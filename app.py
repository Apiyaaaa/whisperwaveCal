from pymongo import MongoClient
import pinecone
import os
from openai import OpenAI
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import logging
import math

# execute every 1 hour


# 修改logging的默认格式
logging.basicConfig(level=logging.INFO, datefmt="%d-%m-%Y %H:%M:%S",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


load_dotenv()

# 一些常量
# 总共需要的新闻数量
news_feed_count = 100
# 聚类的数量
news_cluster_count = 15
# 计算的时间间隔
calculat_interval = 60*60*3

mongo = MongoClient(os.getenv('MONGODB_URI'))
mongo = mongo['NewsSnap']
pinecone.init(api_key=os.getenv(
    'PINECONE_KEY'), environment="gcp-starter")
pinecone_index = pinecone.Index("newssnap-test")
openai = OpenAI()


def get_date_range(period):
    today = datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0)
    if period == 'daily':
        start_date = today - datetime.timedelta(days=2)
    elif period == 'weekly':
        start_date = today - datetime.timedelta(days=7)
    elif period == 'monthly':
        start_date = today - datetime.timedelta(days=30)
    else:
        logging.error(f'period {period} is not valid')
        return None
    return start_date.timestamp(), today.timestamp()


def get_news_by_date(start_date, end_date):
    try:
        pinecone_query = pinecone_index.query(
            vector=[0.1] * 1536, top_k=1000, include_metadata=True, include_values=True, filter={
                'date': {'$gte': start_date, '$lte': end_date}
            })
        if not pinecone_query['matches']:
            return None
    except Exception as e:
        logging.error(e)
        return None
    df = pd.DataFrame([news_item['metadata']
                      for news_item in pinecone_query['matches']])
    df['embedding'] = [news_item['values']
                       for news_item in pinecone_query['matches']]
    df['score'] = [news_item['score']
                   for news_item in pinecone_query['matches']]
    # drop the news that does not have a embedding
    df.dropna(subset=['embedding'], inplace=True)
    df.drop(columns=['source', 'author',
                     'link', 'ai_summary'], inplace=True)
    df.drop_duplicates(subset=['title'], inplace=True)
    return df


def get_news_by_vectors(centers, limit=100, period='daily'):
    if not centers:
        return []
    avg_n_news_per_tag = math.ceil(limit / len(centers))
    start_date, end_date = get_date_range(period)
    if not start_date or not end_date:
        return []
    filters = {
        'date': {'$gte': start_date, '$lte': end_date},
    }
    res = []
    for center in centers:
        news_list = []
        try:
            news = pinecone_index.query(
                vector=center,
                top_k=avg_n_news_per_tag,
                filter=filters,
                include_metadata=True,
                include_values=False
            )
        except Exception as e:
            logging.error(e)
            continue
        if not news['matches']:
            continue
        for news_item in news['matches']:
            temp_news = {
                'id': news_item.get('id', 'N/A'),
                'title': news_item['metadata'].get('title', 'N/A'),
                'link': news_item['metadata'].get('link', 'N/A'),
                'source': news_item['metadata'].get('source', 'N/A'),
                'author': news_item['metadata'].get('author', 'N/A'),
                'date': news_item['metadata'].get('date', 'N/A'),
                'score': news_item['score'],
                'ai_summary': news_item['metadata'].get('ai_summary', 'N/A'),
            }
            news_list.append(temp_news)
        res.append(news_list)
    return res


def best_n_clusters(matrix, n_clusters):
    scores = []
    for n in range(2, n_clusters):
        kmeans = KMeans(n_clusters=n, random_state=42,
                        n_init='auto').fit(matrix)
        score = silhouette_score(matrix, kmeans.labels_)
        scores.append(score)
    return np.argmax(scores) + 2 if scores else 2


def get_cluster(matrix, max_n_clusters=10):
    n_clusters = best_n_clusters(matrix, max_n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0,
                    n_init='auto').fit(matrix)
    centers = kmeans.cluster_centers_
    return centers


def is_calculation_needed(start_date, end_date, period):
    current_time = datetime.datetime.now().timestamp()
    mongo_query = mongo['feeds'].find({
        'start_date': start_date,
        'end_date': end_date,
        'period': period,
        'feed': {'$exists': True}
    }).limit(1)
    query = list(mongo_query)
    if not query:
        return True
    created_at = query[0]['created_at']
    if (current_time - created_at) > calculat_interval:
        return True
    if not query[0]['feed']:
        return True
    return False


def create_tags(title_lists):
    n = 0
    prompt = """
    你的任务是同一聚类中的新闻创建一个标签。
    要求：
    1. 标签不能超过10个字
    2. 标签不用包含新闻的关键词
    3. 标签需要能够代表这一类新闻
    4. 你可以使用抽象的词语
    5. 你除了回复你创建的标签以外不需要做任何事情
    6. 新闻标题会以###分隔
    示例：
    石油交响曲
    娱乐业风波
    灾难纷至沓来
    能量之舞
    """
    while n < 3:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                        "content": prompt},
                    {"role": "user", "content": '###'.join(title_lists)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(e)
            n += 1
            continue
    return None


def tag_news(news_list):
    title_lists = [news_item['title'] for news_item in news_list]
    tag = create_tags(title_lists)
    if tag is None:
        tag = 'N/A'
    return {
        'tag': tag,
        'news': news_list
    }


def save_feed_to_mongo(feed, start_date, end_date, period):
    try:
        length = 0
        for i in feed:
            length += len(i['news'])
        mongo['feeds'].update_one({
            'start_date': start_date,
            'end_date': end_date,
            'period': period
        }, {
            '$set': {
                'feed': feed,
                'created_at': datetime.datetime.now().timestamp(),
                'count': length
            }
        }, upsert=True)
    except Exception as e:
        logging.error(e)
        return None


def main():
    script_start_time = datetime.datetime.now()
    cal_need = ['daily', 'weekly', 'monthly']
    for period in cal_need:
        execute_time = datetime.datetime.now()
        start_date, end_date = get_date_range(period)
        if not is_calculation_needed(start_date, end_date, period):
            logging.info('不需要计算')
            continue
        logging.info(f'开始计算新闻聚类{period}')
        df = get_news_by_date(start_date, end_date)
        if df is None:
            logging.info('没有新闻')
            save_feed_to_mongo([], start_date, end_date, period)
            continue
        centers = get_cluster(
            np.array(df['embedding'].tolist()), news_cluster_count)
        news_list = get_news_by_vectors(
            centers.tolist(), news_feed_count, period)
        logging.info(f'新闻共分类为{len(centers)}类，每类平均{len(news_list[0])}条')
        # 提取title
        res = []
        for news in news_list[0:5]:
            res.append(tag_news(news))
        save_feed_to_mongo(res, start_date, end_date, period)
        logging.info(f'计算完成, 耗时{datetime.datetime.now() - execute_time}')
    logging.info(f'总耗时{datetime.datetime.now() - script_start_time}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f'错误：{e}')

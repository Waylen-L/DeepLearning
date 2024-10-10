import requests
from pathlib import Path
from PIL import Image
import io
#import pandas as pd
from urllib.parse import urlencode
import csv
import concurrent.futures

from config import INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID

tag = '夏コーデ'
save_dir = Path('src/img/male')
media_data_dir = Path('src/media_data/male')

def get_hashtag_media(hashtag_id, access_token, limit=100):
    """
    指定されたタグに関係する図をゲット

    パラメータ:
    hashtag_id: 文字列、タグIDを表示
    access_token: 文字列, APIを訪問するのに必要なアクセストーカン
    limit: 整数, 毎回のリクエストがリターンする最大の結果数, デフォルトが 100

    リターン:
    写真の情報を含めたリスト

    異常:
    APIのリクエストが失敗する場合, 異常をthrow
    """
    all_media = []
    after = None

    while True:
        url = f'https://graph.facebook.com/v12.0/{hashtag_id}/media?fields=id,caption,media_type,media_url,permalink,timestamp&access_token={access_token}&limit={limit}'
        if after:
            url += f'&after={after}'

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # 今のページのデータを結果リストに追加
            all_media.extend(data['data'])

            # さらにデータがあれば, afterを更新してループ
            if 'paging' in data and 'next' in data['paging']:
                after = data['paging']['cursors']['after']
            else:
                break
        except requests.RequestException as e:
            print(f"Error When Get Photo: {e}")
            raise

    return all_media[:100]  # 前のxxx枚の写真だけをリターン

# ディレクトリが存在するのを確保
save_dir.mkdir(parents=True, exist_ok=True)
media_data_dir.mkdir(parents=True, exist_ok=True)

def get_hashtag_id(tag, user_id, access_token):
    # URLリクエスト: APIバージョン、パラメータとアクセストーカンの調査
    url = f'https://graph.facebook.com/v12.0/ig_hashtag_search?user_id={user_id}&q={tag}&access_token={access_token}'

    try:
        # URLリクエストの情報をゲット
        response = requests.get(url)
        # リクエストが成功したかをチェック, なければ異常をthrow
        response.raise_for_status()
        # responseのJSONデータを解析し, 一番目のタグのIDをリターン
        return response.json()['data'][0]['id']
    except requests.RequestException as e:
        print(f"Error When Get Tag ID: {e}")
        raise


def get_media_data(hashtag_id, user_id, access_token):
    base_url = f'https://graph.facebook.com/v12.0/{hashtag_id}/top_media'
    params = {
        'user_id': user_id,
        'fields': 'id,media_type,media_url,permalink,caption,like_count,comments_count',
        'access_token': access_token
    }
    url = f'{base_url}?{urlencode(params)}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['data']
    except requests.RequestException as e:
        print(f"Error When Get Media Data: {e}")
        raise

def filter_media_data(media_data, filter_keywords):
    return [item for item in media_data if not any(keyword.lower() in item.get('caption', '').lower() for keyword in filter_keywords)]

def download_and_save_image(media_url, file_path):
    try:
        media_response = requests.get(media_url)
        media_response.raise_for_status()

        img = Image.open(io.BytesIO(media_response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img.save(file_path, 'JPEG')
        return True
    except requests.RequestException as e:
        print(f"Can Not Download Media: {e}")
    except IOError as e:
        print(f"Can Not Handle Or Save Photo: {e}")
    return False

def save_media_info(media_info, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['id', 'original_id', 'media_type', 'permalink', 'caption', 'like_count', 'comments_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(media_info)

def main():
    try:
        hashtag_id = get_hashtag_id(tag, INSTAGRAM_USER_ID, INSTAGRAM_ACCESS_TOKEN)
        media_data = get_media_data(hashtag_id, INSTAGRAM_USER_ID, INSTAGRAM_ACCESS_TOKEN)
        media_data = filter_media_data(media_data, filter_keywords=['海外', '女', 'ギャル', '可愛い'])
        saved_images = 0
        media_info = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, item in enumerate(media_data):
                if i >= 5:  # 保存する図の数
                    break

                media_url = item['media_url']
                file_name = f"{i + 1}.jpg"
                file_path = save_dir / file_name

                future = executor.submit(download_and_save_image, media_url, file_path)
                if future.result():
                    saved_images += 1
                    print(f"Photo Saved: {file_name}")

                    media_info.append({
                        'id': saved_images,
                        'original_id': item['id'],
                        'media_type': item['media_type'],
                        'permalink': item['permalink'],
                        'caption': item.get('caption', ''),
                        'like_count': item.get('like_count', 0),
                        'comments_count': item.get('comments_count', 0)
                    })

        csv_file_path = media_data_dir / 'media_info_female.csv'
        save_media_info(media_info, csv_file_path)
        print(f"Media Data Saved In: {csv_file_path}")
        print(f"Saved {saved_images} Photo")

        """
        # CSVファイルを読み込む
        media_info_df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(media_info_df)
        """

    except Exception as e:
        print(f"Program Exception: {e}")
        raise

if __name__ == "__main__":
    main()


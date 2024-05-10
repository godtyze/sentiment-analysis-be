import pandas as pd
import googleapiclient.discovery
import os
import regex as re

def get_youtube_comments(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyAxHc64ToHUCmnMAQYzCyngS1k9SKNl64o"  # Use an environment variable for the API key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    comments = []
    page_token = None
    while len(comments) < 500:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Max allowed by YouTube API
            pageToken=page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            text = comment['textDisplay']

            # Regex to match non-English characters
            # if re.match("^[a-zA-Z0-9\s,.'-?!]*$", text):
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                text
             ])

        page_token = response.get('nextPageToken')
        if not page_token:  # Break the loop if there is no next page
            break

    return pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

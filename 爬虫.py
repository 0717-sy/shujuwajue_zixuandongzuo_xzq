import requests
import re
import json
import time
import os
import datetime
import random
import jieba

# 定义薛之谦常用独特词汇，添加到jieba词典
XZQ_WORDS = [
    '丑八怪', '演员', '绅士', '方圆几里', '配合', '分开时',
    '天外来物', '认真的雪', '意外', '刚好', '刚刚好', '动物世界',
    '租购', '天份', '那是什么', '离开北京', '陪你去流浪', '小尖尖',
    '暧昧', '你还要我怎样', '耗尽', '一半', '怪咖', '我好像在哪见过你',
    '像风一样', '我终于成了别人的女人', '我知道你都知道', '爱我的人',
    '谢谢你', '黄色枫叶', '爱的期限', '给我的爱人', '我们的世界', '倾城',
    '红尘女子', '你过得好吗', '马戏小丑', '传说', '我的雅典娜', '未完成的歌',
    '梦开始的原点', '深深爱过你', '星河之役', '流星的眼泪', '续雪', '丢手绢',
    '朋友你们还好吗', '爱情宣判', '苏黎世的从前', '我的Show', '快乐帮', '爱不走',
    '钗头凤', '王子归来', 'OK OK', '其实', '演员', '绅士', '配合', '分开',
    '天外来物', '认真的雪', '意外', '刚好', '刚刚好', '动物世界'
]

# 添加自定义词典
for word in XZQ_WORDS:
    jieba.add_word(word)

# 网易云音乐API爬取歌词
def crawl_xuezhiqian_lyrics():
    
    # 薛之谦专辑发行年份映射
    ALBUM_YEAR_MAP = {
        # 早期 (2006-2012)
        '薛之谦': 2006, '你过得好吗': 2007, '深深爱过你': 2008,
        '未完成的歌': 2009, '传说': 2006, '流星的眼泪': 2007,
        '苏黎世的从前': 2008, '我的show': 2009, '快乐帮': 2009,
        '王子归来': 2010, '几个你': 2011, '几个薛之谦': 2011,
     '续雪': 2012, '认真的雪': 2006,
        # 中期 (2013-2017)
        '意外': 2013, '绎士': 2015, '丑八怪': 2015,
        '初学者': 2016, '演员': 2016, '我好像在哪见过你': 2016,
        '高尚': 2017, '动物世界': 2017, '怒': 2017, '别': 2017,
        '像风一样': 2016, '刚刚好': 2016,
        # 近期 (2018-至今)
        '怒放版': 2018, '天外来物': 2020, '天分': 2021,
        '天份': 2021,
        '尘': 2019, '木偶人': 2019, 
        '天分': 2021, 
        '那是你离开了北京的生活': 2019,
        '租购': 2023,
    }
    
    # 薛之谦的网易云音乐歌手ID
    artist_id = 5781
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://music.163.com/',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }
    
    lyrics_list = []
    all_songs = []
    
    try:
        # 使用网易云音乐API获取歌手热门歌曲
        songs_url = f'https://music.163.com/api/artist/{artist_id}'
        
        response = requests.get(songs_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f'获取歌曲列表失败，状态码: {response.status_code}')
            return lyrics_list
        
        data = response.json()
        
        if 'hotSongs' in data:
            all_songs.extend(data['hotSongs'])

        for offset in range(0, 500, 50):
            try:
                more_url = f'https://music.163.com/api/v1/artist/songs?id={artist_id}&offset={offset}&limit=50&order=time&private_cloud=true&work_type=1'
                more_response = requests.get(more_url, headers=headers, timeout=10)
                more_data = more_response.json()
                if 'songs' in more_data and more_data['songs']:
                    all_songs.extend(more_data['songs'])
                else:
                    break
                time.sleep(0.2)
            except:
                break
        
        # 去重
        seen_ids = set()
        songs = []
        for s in all_songs:
            if s['id'] not in seen_ids:
                seen_ids.add(s['id'])
                songs.append(s)
        
        print(f'\n总共获取到 {len(songs)} 首歌曲（去重后）')
        
        # 处理每首歌曲
        for i, song in enumerate(songs):
            song_id = song['id']
            song_name = song['name']
            
            # 获取专辑信息
            album = song.get('al', {}) or song.get('album', {})
            album_name = album.get('name', '')
            
            # 获取发行时间 - 优先从专辑获取
            publish_time = 0
            if album and album.get('publishTime'):
                publish_time = album.get('publishTime', 0)
            if not publish_time:
                publish_time = song.get('publishTime', 0)
            if not publish_time and album:
                # 尝试从album对象获取
                publish_time = album.get('picId', 0)  # 有时picId包含时间信息
            
            release_date = ''
            if publish_time and publish_time > 0:
                try:
                    release_date = datetime.datetime.fromtimestamp(publish_time / 1000).strftime('%Y-%m-%d')
                except:
                    release_date = ''
            
            # 如果没有发行时间，根据专辑名称查找
            if not release_date and album_name:
                for album_key, year in ALBUM_YEAR_MAP.items():
                    if album_key in album_name or album_name in album_key:
                        release_date = f'{year}-01-01'
                        break
            
            print(f'\n[{i+1}/{len(songs)}] 处理歌曲: {song_name}')
            
            # 处理Live版本和+号歌名，跳过相关歌曲
            skip_keywords = ['伴奏', 'Instrumental', '吉他','Live', '现场', '+']
            if any(kw in song_name for kw in skip_keywords):
                print(f'  跳过: {song_name}')
                continue
            
            # 清理歌名后缀
            original_song_name = song_name
            song_name = re.sub(r'\s*\(伴奏\)|\s*\[伴奏\]', '', song_name)
            song_name = re.sub(r'\s*\([^)]*翻唱[^)]*\)', '', song_name)
            song_name = song_name.strip()
            
            # 获取歌词
            lyric_url = f'https://music.163.com/api/song/lyric?id={song_id}&lv=1&tv=1'
            
            try:
                lyric_response = requests.get(lyric_url, headers=headers, timeout=10)
                lyric_data = lyric_response.json()
                
                if 'lrc' in lyric_data and 'lyric' in lyric_data['lrc']:
                    raw_lyric = lyric_data['lrc']['lyric']
                    
                    # 清理歌词：移除时间戳
                    clean_lyric = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', raw_lyric)
                    clean_lyric = re.sub(r'\[\d{2}:\d{2}\]', '', clean_lyric)
                    
                    # 移除元数据行和非歌词部分
                    lines = clean_lyric.split('\n')
                    lyric_lines = []
                    
                    metadata_patterns = [
                        r'^作词', r'^作曲', r'^编曲', r'^制作', r'^混音', r'^母带',
                        r'^录音', r'^和声', r'^吉他', r'^贝斯', r'^鼓', r'^键盘',
                        r'^弦乐', r'^出品', r'^发行', r'^OP', r'^SP', r'^by:',
                        r'^词：', r'^曲：', r'^演唱', r'^原唱', r'^翻唱',
                        r'^\[', r'^\]', r'^\(', r'^\)',  # 括号开头的标记
                        r'^\[.*?\]$',  # 完整的方括号标记如[Intro] [Verse] [Chorus]
                        r'^.*?\[.*?\].*?$',  # 包含方括号的行
                        r'^\d+:\d+',  # 时间戳
                        r'^\d+\s*秒',  # 秒数标记
                        r'^\s*\*\s*$',  # 纯星号行
                        r'^\s*#.*?#\s*$',  # 前后有#的标记
                    ]
                    
                    # 用于标记是否在非歌词部分
                    in_non_lyric_section = False
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 检查是否为非歌词标记
                        is_non_lyric_marker = False
                        for p in metadata_patterns:
                            if re.match(p, line):
                                is_non_lyric_marker = True
                                break
                        
                        # 检查是否为常见的非歌词标记
                        non_lyric_markers = ['[Intro]', '[Verse]', '[Chorus]', '[Bridge]', '[Interlude]', '[Outro]', '[Pre-Chorus]', '[Hook]', '[Solo]', '[Break]', '[Refrain]', '[Post-Chorus]', '[Intro]', '[Outro]', '[Interlude]']
                        if any(marker in line for marker in non_lyric_markers):
                            is_non_lyric_marker = True
                        
                        if is_non_lyric_marker:
                            in_non_lyric_section = True
                            continue
                        
                        # 如果在非歌词部分，跳过内容直到遇到新的标记或歌词开始
                        if in_non_lyric_section and not is_non_lyric_marker:
                            in_non_lyric_section = False
                        
                        # 跳过纯英文制作人员信息
                        try:
                            if re.match(r'^[A-Za-z\\s]+:', line):
                                continue
                        except re.error:
                            continue
                        
                        # 跳过过短的行（可能是标记）
                        try:
                            if len(line) < 3 and re.match(r'^[A-Za-z\s\d\W]*$', line):
                                continue
                        except re.error:
                            continue
                        
                        # 添加歌词行
                        lyric_lines.append(line)
                    
                    clean_lyric = '\n'.join(lyric_lines)
                    
                    # 进一步清理歌词，去除可能遗漏的非歌词部分
                    # 去除[开头但没有歌词内容的行
                    final_lines = []
                    for line in clean_lyric.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('[') and not line.endswith(']') and len(line) > 1:
                            final_lines.append(line)
                    clean_lyric = '\n'.join(final_lines)
                    
                    # 确保歌词有效
                    if len(clean_lyric) < 30:
                        print(f'歌词过短')
                        continue
                    
                    # 提取作词作曲信息
                    lyricist = ''
                    composer = ''
                    
                    lyricist_match = re.search(r'作词[：:](.*?)(?:\n|$)', raw_lyric)
                    if lyricist_match:
                        lyricist = lyricist_match.group(1).strip()
                    
                    composer_match = re.search(r'作曲[：:](.*?)(?:\n|$)', raw_lyric)
                    if composer_match:
                        composer = composer_match.group(1).strip()
                    
                    # 使用jieba进行分词
                    words = jieba.lcut(clean_lyric)
                    # 过滤掉空字符串和单个字符
                    words = [word for word in words if word.strip() and len(word) > 1]
                    
                    song_data = {
                        'song_name': song_name,
                        'artist': '薛之谦',
                        'lyric': clean_lyric,
                        'words': words,  # 添加分词结果
                        'lyricist': lyricist,
                        'composer': composer,
                        'release_date': release_date,
                        'album': album_name
                    }
                    
                    lyrics_list.append(song_data)
                    print(f'成功获取歌词，长度: {len(clean_lyric)}字符')
                else:
                    print(f'无歌词数据')
                
            except Exception as e:
                print(f'获取歌词失败: {e}')
            
            # 控制请求速度
            time.sleep(0.3)
        
    except Exception as e:
        print(f'爬取失败: {e}')
        import traceback
        traceback.print_exc()

    return lyrics_list

# 清理歌名，移除括号内容和合作歌手信息，保留Live版本但移除后缀
def clean_song_name(song_name):
    # 移除合作歌手信息（斜杠后的内容）
    song_name = re.sub(r'\s*/[^/]+$', '', song_name)
    # 移除"薛之谦"字样
    song_name = re.sub(r'\s*薛之谦\s*', '', song_name)
    # 移除各种括号内容，但保留Live标记
    # 先处理非Live的括号内容
    song_name = re.sub(r'\s*\((?!Live)[^)]*\)\s*', '', song_name)
    song_name = re.sub(r'\s*\[(?!Live)[^\]]*\]\s*', '', song_name)
    # 移除Live版本的后缀，但保留Live标记
    song_name = re.sub(r'\s*\(Live版?\)|\s*\[Live版?\]|\s*现场版?|\s*Live', '(Live)', song_name)
    # 移除伴奏标记
    song_name = re.sub(r'\s*\(伴奏\)|\s*\[伴奏\]', '', song_name)
    # 移除翻唱标记
    song_name = re.sub(r'\s*\([^)]*翻唱[^)]*\)', '', song_name)
    # 移除其他版本标记
    song_name = re.sub(r'\s*(Version|Remix|Acoustic|Studio|Karaoke|伴奏版)\s*', '', song_name, flags=re.IGNORECASE)
    # 移除多余空格
    song_name = song_name.strip()
    return song_name

# 薛之谦歌曲时期分类
def get_song_period(release_date):
    """根据发行时间判断时期：
    2006-2012: 早期
    2013-2017: 中期 
    2018-至今: 近期
    """
    if not release_date:
        return '2018-至今'
    
    try:
        # 提取年份
        if '-' in release_date:
            year = int(release_date.split('-')[0])
        else:
            year = int(release_date[:4])
        
        if 2006 <= year <= 2012:
            return '2006-2012'
        elif 2013 <= year <= 2017:
            return '2013-2017'
        elif year >= 2018:
            return '2018-至今'
        else:
            return '2018-至今'
    except:
        # 日期解析失败，默认返回近期
        return '2018-至今'

# 主函数
if __name__ == '__main__':
    lyrics_data = crawl_xuezhiqian_lyrics()
    
    # 清理歌名
    for song in lyrics_data:
        song['song_name'] = clean_song_name(song['song_name'])
    
    # 去重，保留同歌名的第一首
    unique_lyrics = []
    seen = set()
    for song in lyrics_data:
        if song['song_name'] not in seen:
            seen.add(song['song_name'])
            unique_lyrics.append(song)
    
    # 添加时期分类
    for song in unique_lyrics:
        song['period'] = get_song_period(song.get('release_date', ''))
    
    # 统计各时期歌曲数量
    period_count = {'2006-2012': 0, '2013-2017': 0, '2018-至今': 0}
    for song in unique_lyrics:
        period_count[song['period']] += 1
    
    for period, count in period_count.items():
        print(f'{period}: {count} 首')
    
    # 创建data目录（如果不存在）
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'薛之谦歌词分析','utils', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存到JSON文件
    output_file = os.path.join(data_dir, 'xuezhiqian.json')
    print(f'\n准备保存 {len(unique_lyrics)} 首歌曲到 {output_file}')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_lyrics, f, ensure_ascii=False, indent=2)
        print(f'保存完成，{os.path.exists(output_file)}')
    except Exception as e:
        print(f'保存文件时出错: {e}')
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        output_file = os.path.join(parent_dir, 'utils', 'data', 'xuezhiqian.json')
        print(f'尝试使用备用路径: {output_file}')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_lyrics, f, ensure_ascii=False, indent=2)
        print(f'备用路径保存完成，{os.path.exists(output_file)}')
    
    # 检查是否满足要求
    total_songs = len(unique_lyrics)
    early_songs = period_count['2006-2012']
    middle_songs = period_count['2013-2017']
    recent_songs = period_count['2018-至今']
    print(f'\n共获取到 {len(unique_lyrics)} 首歌曲。')
    print(f'数据已保存到：{output_file}')

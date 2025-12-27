import jieba
import jieba.analyse
import json
import re
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from datetime import datetime
import seaborn as sns
from snownlp import SnowNLP
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class LyricsAnalyzer:
    def __init__(self, preprocessed_data_path=None):
        self.data = []
        if preprocessed_data_path and os.path.exists(preprocessed_data_path):
            with open(preprocessed_data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        self.preprocessor = None
    
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def load_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
    
    def is_likely_name(self, word):
        # 简单规则：中文姓名通常为2-4个字符，且为中文字符
        if 2 <= len(word) <= 4:
            # 检查是否全为中文字符
            for char in word:
                if not (0x4e00 <= ord(char) <= 0x9fff):  # 中文字符范围
                    return False
            # 常见姓名用字
            common_name_chars = {'伟', '芳', '娜', '秀英', '敏', '静', '丽', '强', '磊', '军', '洋', '勇', '艳', '杰', '娟', '涛', '明', '超', '秀兰', '霞', '平', '刚', '桂英'}
            if any(char in common_name_chars for char in word):
                return True
            # 检查是否为常见的单字姓氏
            common_surnames = {'王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴', '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗', '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧', '程', '曹', '袁', '邓', '许', '傅', '沉', '曾', '彭', '吕', '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎', '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜', '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆', '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史', '顾', '侯', '邵', '孟', '龙', '万', '段', '雷', '钱', '汤', '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文'}
            if word[0] in common_surnames:
                return True
        return False

    def word_frequency_analysis(self, top_k=50):
        all_words = []
        for song in self.data:
            words = song.get('words', [])
            all_words.extend(words)
            
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(top_k)
            
        # 按词性分类高频词
        nouns = []
        verbs = []
        adjectives = []
            
        for song in self.data:
            nouns.extend(song.get('nouns', []))
            verbs.extend(song.get('verbs', []))
            adjectives.extend(song.get('adjectives', []))
            
        noun_freq = Counter(nouns)
        verb_freq = Counter(verbs)
        adj_freq = Counter(adjectives)
            
        # 分离否定词
        negation_words = self.negation_analysis()
        negation_freq = Counter()
        for song_data in negation_words['song_level']:
            negation_freq.update(song_data['negation_words'])
            
        return {
            'all_words': top_words,
            'nouns': noun_freq.most_common(top_k),
            'verbs': verb_freq.most_common(top_k),
            'adjectives': adj_freq.most_common(top_k),
            'negations': negation_freq.most_common(top_k)
        }
        
    def lyrics_length_analysis(self):
        length_data = []
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            word_count = len(song.get('words', []))
            char_count = len(lyric.replace('\n', '').replace(' ', ''))
            length_data.append({
                'song_name': song.get('song_name', ''),
                'word_count': word_count,
                'char_count': char_count,
                'release_date': song.get('release_date', '')
            })
            
        # 按时间分析长度变化
        df = pd.DataFrame(length_data)
        df['year'] = df['release_date'].apply(lambda x: x[:4] if x and len(x) >= 4 else 'Unknown')
            
        return df
        
    def vocabulary_richness_analysis(self):
        total_types = set()
        total_tokens = 0
            
        for song in self.data:
            words = song.get('words', [])
            total_types.update(words)
            total_tokens += len(words)
            
        if total_tokens == 0:
            return 0, 0, 0
            
        ttr = len(total_types) / total_tokens  # Type-Token Ratio
        return ttr, len(total_types), total_tokens
        
    def metaphor_analysis_v1(self):
        metaphor_keywords = ['像', '如', '似', '仿佛', '好像', '犹如', '如同', '好比', '宛如', '恰似', '一般', '一样']
        metaphor_patterns = []
            
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            lines = lyric.split('\n')
                
            for line in lines:
                for keyword in metaphor_keywords:
                    if keyword in line:
                        # 提取比喻前后的词
                        parts = line.split(keyword)
                        if len(parts) >= 2:
                            # 获取比喻前的词（通常是名词或代词）
                            before_parts = parts[0].split()[-5:] if len(parts[0].split()) >= 5 else parts[0].split()
                            # 获取比喻后的词（通常是名词或描述）
                            after_parts = parts[1].split()[:5] if len(parts[1].split()) >= 5 else parts[1].split()
                            metaphor_patterns.append({
                                'song': song.get('song_name', ''),
                                'metaphor_word': keyword,
                                'before': ' '.join(before_parts),
                                'after': ' '.join(after_parts),
                                'full_sentence': line
                            })
            
        return metaphor_patterns
        
    def sentence_complexity_analysis_v1(self):
        sentence_data = []
            
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            lines = [line for line in lyric.split('\n') if line.strip()]
                
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            avg_line_length = np.mean(line_lengths) if line_lengths else 0
            std_line_length = np.std(line_lengths) if line_lengths else 0
                
            # 统计标点符号
            punctuation_count = {
                '，': lyric.count('，'),
                '。': lyric.count('。'),
                '！': lyric.count('！'),
                '？': lyric.count('？'),
                '；': lyric.count('；'),
                '：': lyric.count('：'),
                '、': lyric.count('、'),
                '—': lyric.count('—') + lyric.count('—'),  # 长破折号
            }
                
            sentence_data.append({
                'song_name': song.get('song_name', ''),
                'avg_line_length': avg_line_length,
                'std_line_length': std_line_length,
                'total_lines': len(lines),
                'punctuation': punctuation_count
            })
            
        return sentence_data
        
    def unique_word_analysis(self, comparison_data_path=None):
        # 获取薛之谦的词频
        xzq_words = []
        for song in self.data:
            xzq_words.extend(song.get('words', []))
        xzq_word_freq = Counter(xzq_words)
            
        if comparison_data_path and os.path.exists(comparison_data_path):
            with open(comparison_data_path, 'r', encoding='utf-8') as f:
                comp_data = json.load(f)
                
            comp_words = []
            for song in comp_data:
                if 'words' in song:
                    comp_words.extend(song['words'])
                else:
                    # 如果没有预处理，需要进行分词
                    import jieba
                    comp_words.extend(jieba.lcut(song.get('lyric', '')))
            comp_word_freq = Counter(comp_words)
                
            # 计算TF-IDF找出独特词
            all_words = set(xzq_word_freq.keys()) | set(comp_word_freq.keys())
            xzq_unique = {}
            comp_unique = {}
                
            for word in all_words:
                xzq_tf = xzq_word_freq.get(word, 0)
                comp_tf = comp_word_freq.get(word, 0)
                    
                if xzq_tf > comp_tf and xzq_tf > 2:  # 薛之谦使用更多
                    xzq_unique[word] = xzq_tf - comp_tf
                elif comp_tf > xzq_tf and comp_tf > 2:  # 对方使用更多
                    comp_unique[word] = comp_tf - xzq_tf
                
            return {
                'xzq_unique': sorted(xzq_unique.items(), key=lambda x: x[1], reverse=True)[:50],
                'comp_unique': sorted(comp_unique.items(), key=lambda x: x[1], reverse=True)[:50]
            }
            
        return {'xzq_unique': xzq_word_freq.most_common(50)}
        
    # 修辞与句法分析
    def metaphor_analysis(self):
        metaphor_keywords = ['像', '如', '似', '仿佛', '好像', '犹如', '如同', '好比', '宛如', '恰似']
        metaphor_patterns = []
            
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            lines = lyric.split('\n')
                
            for line in lines:
                for keyword in metaphor_keywords:
                    if keyword in line:
                        parts = line.split(keyword)
                        if len(parts) >= 2:
                            before = parts[0].split()[-3:] if len(parts[0].split()) >= 3 else parts[0].split()
                            after = parts[1].split()[:3] if len(parts[1].split()) >= 3 else parts[1].split()
                            metaphor_patterns.append({
                                'song': song.get('song_name', ''),
                                'metaphor_word': keyword,
                                'before': ' '.join(before),
                                'after': ' '.join(after),
                                'full_sentence': line
                            })
            
        return metaphor_patterns
        
    def negation_analysis(self):
        negation_words = ['不', '没', '无', '别', '勿', '未', '非', '不是', '没有', '不要', '不能', '不会', '不要求']
        negation_count = Counter()
        song_negation = []
            
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            words = song.get('words', [])
                
            negation_in_song = [w for w in words if w in negation_words]
            negation_count.update(negation_in_song)
                
            song_negation.append({
                'song_name': song.get('song_name', ''),
                'negation_count': len(negation_in_song),
                'negation_words': negation_in_song
            })
            
        return {
            'overall_freq': negation_count.most_common(),
            'song_level': song_negation
        }
        
    def sentence_complexity_analysis(self):
        sentence_data = []
            
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            lines = [line for line in lyric.split('\n') if line.strip()]
                
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            avg_line_length = np.mean(line_lengths) if line_lengths else 0
            std_line_length = np.std(line_lengths) if line_lengths else 0
                
            # 统计标点符号
            punctuation_count = {
                '，': lyric.count('，'),
                '。': lyric.count('。'),
                '！': lyric.count('！'),
                '？': lyric.count('？'),
                '；': lyric.count('；'),
                '：': lyric.count('：'),
                '、': lyric.count('、'),
                '—': lyric.count('—') + lyric.count('—'),  # 长破折号
            }
                
            sentence_data.append({
                'song_name': song.get('song_name', ''),
                'avg_line_length': avg_line_length,
                'std_line_length': std_line_length,
                'total_lines': len(lines),
                'punctuation': punctuation_count
            })
            
        return sentence_data

    def sentiment_analysis(self):
        """情感分析（基于snownlp，改进公式使结果更合理，偏向消极）"""
        sentiment_data = []
            
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            if lyric.strip():
                try:
                    s = SnowNLP(lyric)
                    sentiment_score = s.sentiments

                    # 薛之谦的歌词偏向消极，所以需要调整情感值
                    lyric_lower = lyric.lower()
                        
                    # 检查是否有负面情感关键词
                    negative_keywords = ['不', '没', '无', '恨', '痛', '伤', '泪', '哭', '错', '悔', '恨', '怨', '怒', '悲', '愁', '哀', '苦', '痛', '难', '烦', '厌', '惧', '怕', '绝望', '放弃', '痛恨', '讨厌', '嫌弃', '难过', '伤心', '失望', '沮丧', '孤独', '寂寞', '痛苦', '折磨', '分手', '离别', '背叛', '辜负', '心碎', '眼泪', '眼泪', '眼泪', '眼泪', '眼泪']
                    negative_count = sum(1 for keyword in negative_keywords if keyword in lyric_lower)
                        
                    # 检查是否有正面情感关键词
                    positive_keywords = ['爱', '喜', '笑', '乐', '好', '美', '甜', '暖', '幸', '福', '欢', '悦', '赞', '好', '妙', '棒', '赞', '美', '爱', '恋', '甜', '心', '情', '喜欢', '开心', '快乐', '幸福', '温暖', '感动', '美好', '希望', '阳光', '微笑', '拥抱', '期待', '感谢', '感恩', '珍惜', '美好', '美好', '美好', '美好', '美好']
                    positive_count = sum(1 for keyword in positive_keywords if keyword in lyric_lower)
                        
                    # 根据关键词数量调整情感值
                    total_keywords = positive_count + negative_count
                    if total_keywords > 0:
                        # 计算情感偏向
                        sentiment_adjustment = (positive_count - negative_count) / total_keywords
                        sentiment_score = max(0.1, min(0.9, sentiment_score + sentiment_adjustment * 0.4))
                            
                        if negative_count > positive_count:
                            sentiment_score = sentiment_score * 0.8  # 进一步降低，偏向消极
                    else:
                        sentiment_score = sentiment_score * 0.9
                            
                    # 确保情感值在合理范围内
                    sentiment_score = max(0.1, min(0.9, sentiment_score))
                                
                except:
                    sentiment_score = 0.3  # 默认偏向消极
            else:
                sentiment_score = 0.3  # 默认偏向消极
                    
            sentiment_data.append({
                'song_name': song.get('song_name', ''),
                'sentiment_score': sentiment_score,
                'release_date': song.get('release_date', ''),
                'period': song.get('period', '')
            })
                
        return sentiment_data
        
    def temporal_sentiment_analysis(self):
        sentiment_data = self.sentiment_analysis()
        df = pd.DataFrame(sentiment_data)
            
        # 按年份分组
        df['year'] = df['release_date'].apply(lambda x: x[:4] if x and len(x) >= 4 else 'Unknown')
        yearly_sentiment = df[df['year'] != 'Unknown'].groupby('year')['sentiment_score'].mean().reset_index()
            
        return yearly_sentiment, df
        
    def period_comparison_analysis(self):
        # 按时期分组
        period_data = {
            '2006-2012': [],
            '2013-2017': [],
            '2018-至今': []
        }
        
        for song in self.data:
            period = song.get('period', '2018-至今')  # 默认为近期
            period_data[period].append(song)
        
        # 计算各时期的统计信息
        period_stats = {}
        for period, songs in period_data.items():
            if songs:
                # 情感分析
                sentiments = []
                for song in songs:
                    lyric = song.get('cleaned_lyric', '')
                    if lyric.strip():
                        try:
                            s = SnowNLP(lyric)
                            sentiment = s.sentiments

                            lyric_lower = lyric.lower()
                            
                            # 检查是否有负面情感关键词
                            negative_keywords = ['不', '没', '无', '恨', '痛', '伤', '泪', '哭', '错', '悔', '恨', '怨', '怒', '悲', '愁', '哀', '苦', '痛', '难', '烦', '厌', '惧', '怕', '绝望', '放弃', '痛恨', '讨厌', '嫌弃', '难过', '伤心', '失望', '沮丧', '孤独', '寂寞', '痛苦', '折磨', '分手', '离别', '背叛', '辜负', '心碎', '眼泪', '眼泪', '眼泪', '眼泪', '眼泪']
                            negative_count = sum(1 for keyword in negative_keywords if keyword in lyric_lower)
                            
                            # 检查是否有正面情感关键词
                            positive_keywords = ['爱', '喜', '笑', '乐', '好', '美', '甜', '暖', '幸', '福', '欢', '悦', '赞', '好', '妙', '棒', '赞', '美', '爱', '恋', '甜', '心', '情', '喜欢', '开心', '快乐', '幸福', '温暖', '感动', '美好', '希望', '阳光', '微笑', '拥抱', '期待', '感谢', '感恩', '珍惜', '美好', '美好', '美好', '美好', '美好']
                            positive_count = sum(1 for keyword in positive_keywords if keyword in lyric_lower)
                            
                            # 根据关键词数量调整情感值
                            total_keywords = positive_count + negative_count
                            if total_keywords > 0:
                                # 计算情感偏向
                                sentiment_adjustment = (positive_count - negative_count) / total_keywords
                                sentiment = max(0.1, min(0.9, sentiment + sentiment_adjustment * 0.4))
                                if negative_count > positive_count:
                                    sentiment = sentiment * 0.8
                            else:
                                sentiment = sentiment * 0.9
                            sentiment = max(0.1, min(0.9, sentiment))
                            
                            sentiments.append(sentiment)
                        except:
                            sentiments.append(0.3)  # 默认偏向消极
                    else:
                        sentiments.append(0.3)  # 默认偏向消极
                    
                # 计算统计值
                period_stats[period] = {
                    'song_count': len(songs),
                    'avg_sentiment': np.mean(sentiments) if sentiments else 0.5,
                    'std_sentiment': np.std(sentiments) if len(sentiments) > 1 else 0,
                    'avg_word_count': np.mean([len(song.get('words', [])) for song in songs]) if songs else 0,
                    'avg_char_count': np.mean([len(song.get('cleaned_lyric', '')) for song in songs]) if songs else 0,
                    'vocabulary_richness': self._calculate_vocabulary_richness(songs)
                }
            else:
                period_stats[period] = {
                    'song_count': 0,
                    'avg_sentiment': 0.5,
                    'std_sentiment': 0,
                    'avg_word_count': 0,
                    'avg_char_count': 0,
                    'vocabulary_richness': 0
                }
            
        return period_stats, period_data
        
    def _calculate_vocabulary_richness(self, songs):
        all_words = []
        for song in songs:
            all_words.extend(song.get('words', []))
        
        if not all_words:
            return 0
        
        unique_words = set(all_words)
        return len(unique_words) / len(all_words)
    
    def random_50_songs_sentiment_analysis_divided(self):
        import random
        from snownlp import SnowNLP
        
        # 过滤掉没有歌词的歌曲
        valid_songs = [song for song in self.data if song.get('cleaned_lyric', '').strip()]
        
        # 随机抽取50首歌
        if len(valid_songs) > 50:
            selected_songs = random.sample(valid_songs, 50)
        else:
            selected_songs = valid_songs
        
        # 对每首歌进行情感分析
        results = []
        for song in selected_songs:
            lyric = song.get('cleaned_lyric', '')
            if lyric.strip():
                try:
                    s = SnowNLP(lyric)
                    sentiment = s.sentiments
                    lyric_lower = lyric.lower()
                    negative_keywords = ['不', '没', '无', '恨', '痛', '伤', '泪', '哭', '错', '悔', '恨', '怨', '怒', '悲', '愁', '哀', '苦', '痛', '难', '烦', '厌', '惧', '怕', '绝望', '放弃', '痛恨', '讨厌', '嫌弃', '难过', '伤心', '失望', '沮丧', '孤独', '寂寞', '痛苦', '折磨', '分手', '离别', '背叛', '辜负', '心碎', '眼泪', '眼泪', '眼泪', '眼泪', '眼泪']
                    negative_count = sum(1 for keyword in negative_keywords if keyword in lyric_lower)
                    positive_keywords = ['爱', '喜', '笑', '乐', '好', '美', '甜', '暖', '幸', '福', '欢', '悦', '赞', '好', '妙', '棒', '赞', '美', '爱', '恋', '甜', '心', '情', '喜欢', '开心', '快乐', '幸福', '温暖', '感动', '美好', '希望', '阳光', '微笑', '拥抱', '期待', '感谢', '感恩', '珍惜', '美好', '美好', '美好', '美好', '美好']
                    positive_count = sum(1 for keyword in positive_keywords if keyword in lyric_lower)
                    total_keywords = positive_count + negative_count
                    if total_keywords > 0:
                        sentiment_adjustment = (positive_count - negative_count) / total_keywords
                        sentiment = max(0.1, min(0.9, sentiment + sentiment_adjustment * 0.4))
                        if negative_count > positive_count:
                            sentiment = sentiment * 0.8 
                    else:
                        sentiment = sentiment * 0.9
                    sentiment = max(0.1, min(0.9, sentiment))
                    
                    # 为情感值分类
                    if sentiment > 0.6:
                        sentiment_label = '积极'
                        positive_score = sentiment
                        negative_score = 1 - sentiment
                    elif sentiment < 0.4:
                        sentiment_label = '消极'
                        positive_score = sentiment
                        negative_score = 1 - sentiment
                    else:
                        sentiment_label = '中性'
                        positive_score = sentiment
                        negative_score = 1 - sentiment
                
                except Exception as e:
                    sentiment = 0.3  # 默认偏向消极
                    sentiment_label = '中性'
                    positive_score = 0.3
                    negative_score = 0.7
            else:
                sentiment = 0.5
                sentiment_label = '中性'
                positive_score = 0.5
                negative_score = 0.5
            
            results.append({
                'song_name': song.get('song_name', 'Unknown'),
                'sentiment_score': sentiment,
                'sentiment_label': sentiment_label,  # 添加情感标签
                'positive_score': positive_score,
                'negative_score': negative_score,
                'period': song.get('period', 'Unknown'),
                'lyric_length': len(lyric)
            })

        divided_results = []
        group_size = 10
        for i in range(0, min(50, len(results)), group_size):
            group = results[i:i+group_size]
            divided_results.append(group)
        
        return divided_results
    def lda_topic_modeling(self, n_topics=5, max_features=1000):
        documents = []
        song_names = []
        for song in self.data:
            words = song.get('words', [])
            filtered_words = [word for word in words if not self.is_likely_name(word)]
            if filtered_words:
                doc = ' '.join(filtered_words)
                documents.append(doc)
                song_names.append(song.get('song_name', ''))
        
        if not documents:
            return None, None, None
        
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r'(?u)\b\w+\b')
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # LDA模型
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
        lda.fit(doc_term_matrix)
        
        # 获取主题词
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-20:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            weights = topic[top_words_idx].tolist()
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weights': weights
            })
        
        # 为每首歌分配主题
        doc_topic_dist = lda.transform(doc_term_matrix)
        song_topics = []
        for i, (song_name, topic_dist) in enumerate(zip(song_names, doc_topic_dist)):
            dominant_topic = np.argmax(topic_dist)
            song_topics.append({
                'song_name': song_name,
                'dominant_topic': int(dominant_topic),
                'topic_distribution': topic_dist.tolist()
            })
        
        return topics, song_topics, lda
    
    def co_occurrence_network_analysis(self, top_k=100, window_size=5):
        # 构建词汇共现矩阵
        all_words = []
        for song in self.data:
            words = [w for w in song.get('words', []) if len(w) > 1]  # 只保留长度大于1的词
            all_words.extend(words)
        
        # 计算高频词
        word_freq = Counter(all_words)
        top_words = set([word for word, freq in word_freq.most_common(top_k)])
        
        # 构建共现网络
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for song in self.data:
            words = [w for w in song.get('words', []) if w in top_words]
            for i, word1 in enumerate(words):
                for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                    if i != j:
                        word2 = words[j]
                        co_occurrence[word1][word2] += 1
        
        # 构建NetworkX图
        G = nx.Graph()
        for word1 in co_occurrence:
            for word2 in co_occurrence[word1]:
                weight = co_occurrence[word1][word2]
                if weight > 1:  # 只添加权重大于1的边
                    G.add_edge(word1, word2, weight=weight)

        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            clustering = nx.clustering(G)
            
            return {
                'graph': G,
                'centrality': centrality,
                'betweenness': betweenness,
                'clustering': clustering,
                'co_occurrence_matrix': co_occurrence
            }
        else:
            return None
    def style_evolution_analysis(self):
        """风格演进分析"""
        evolution_data = []
        
        for song in self.data:
            lyric = song.get('cleaned_lyric', '')
            words = song.get('words', [])
            
            # 情感值
            if lyric.strip():
                try:
                    s = SnowNLP(lyric)
                    sentiment = s.sentiments
                    lyric_lower = lyric.lower()
                    negative_keywords = ['不', '没', '无', '恨', '痛', '伤', '泪', '哭', '错', '悔', '恨', '怨', '怒', '悲', '愁', '哀', '苦', '痛', '难', '烦', '厌', '惧', '怕', '绝望', '放弃', '痛恨', '讨厌', '嫌弃', '难过', '伤心', '失望', '沮丧', '孤独', '寂寞', '痛苦', '折磨', '分手', '离别', '背叛', '辜负', '心碎', '眼泪', '眼泪', '眼泪', '眼泪', '眼泪']
                    negative_count = sum(1 for keyword in negative_keywords if keyword in lyric_lower)
                    positive_keywords = ['爱', '喜', '笑', '乐', '好', '美', '甜', '暖', '幸', '福', '欢', '悦', '赞', '好', '妙', '棒', '赞', '美', '爱', '恋', '甜', '心', '情', '喜欢', '开心', '快乐', '幸福', '温暖', '感动', '美好', '希望', '阳光', '微笑', '拥抱', '期待', '感谢', '感恩', '珍惜', '美好', '美好', '美好', '美好', '美好']
                    positive_count = sum(1 for keyword in positive_keywords if keyword in lyric_lower)
                    total_keywords = positive_count + negative_count
                    if total_keywords > 0:
                        sentiment_adjustment = (positive_count - negative_count) / total_keywords
                        sentiment = max(0.1, min(0.9, sentiment + sentiment_adjustment * 0.4))
                        if negative_count > positive_count:
                            sentiment = sentiment * 0.8 
                    else:
                        sentiment = sentiment * 0.9
                    sentiment = max(0.1, min(0.9, sentiment))
                except:
                    sentiment = 0.3  
            else:
                sentiment = 0.3 

            unique_words = len(set(words)) if words else 0
            total_words = len(words) if words else 1
            richness = unique_words / total_words if total_words > 0 else 0
            
            # 根据发行年份确定时期
            release_date = song.get('release_date', '')
            if release_date and len(release_date) >= 4:
                year = int(release_date[:4])
                if year <= 2012:
                    period = '早期 (2006-2012)'
                elif 2013 <= year <= 2017:
                    period = '中期 (2013-2017)'
                else:
                    period = '近期 (2018至今)'
            else:
                period = 'Unknown'
            
            evolution_data.append({
                'song_name': song.get('song_name', ''),
                'release_date': song.get('release_date', ''),
                'sentiment': sentiment,
                'vocabulary_richness': richness,
                'word_count': len(words),
                'period': period  # 使用新确定的时期
            })
        
        df = pd.DataFrame(evolution_data)
        df['year'] = df['release_date'].apply(lambda x: x[:4] if x and len(x) >= 4 else 'Unknown')
        
        # 按时期统计
        period_stats = df[df['period'] != 'Unknown'].groupby('period').agg({
            'sentiment': 'mean',
            'vocabulary_richness': 'mean',
            'word_count': 'mean'
        }).reset_index()
        
        return df, period_stats
    
    def new_word_emergence_analysis(self):
        # 按时期分组歌词
        period_groups = defaultdict(list)
        for song in self.data:
            period = song.get('period', 'Unknown')
            words = song.get('words', [])
            period_groups[period].extend(words)
        
        # 计算各时期词频
        period_word_freq = {}
        for period, words in period_groups.items():
            period_word_freq[period] = Counter(words)
        
        # 找出每个时期特有的高频词
        period_unique_words = {}
        all_periods = list(period_groups.keys())
        
        for target_period in all_periods:
            target_freq = period_word_freq[target_period]
            other_periods = [p for p in all_periods if p != target_period]
            combined_other_freq = Counter()
            for other_period in other_periods:
                combined_other_freq.update(period_word_freq[other_period])
            
            # 计算相对频率
            unique_words = []
            for word, freq in target_freq.most_common(50):
                if len(word) > 1 and freq > 2:  # 只考虑长度大于1且出现次数大于2的词
                    other_freq = combined_other_freq.get(word, 0)
                    if freq > other_freq * 2:  # 目标时期频率是其他时期的2倍以上
                        unique_words.append((word, freq, other_freq))
            
            period_unique_words[target_period] = unique_words[:20]
        
        return period_unique_words

if __name__ == '__main__':
    analyzer = LyricsAnalyzer()
    possible_paths = [
        '薛之谦歌词分析/utils/data/preprocessed_lyrics.json',  # 从项目根目录运行
        '../utils/data/preprocessed_lyrics.json'  # 从utils目录运行
    ]
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'preprocessed_lyrics.json')
        if not os.path.exists(data_path):
            data_path = '../utils/data/preprocessed_lyrics.json'
    if os.path.exists(data_path):
        analyzer.load_data(data_path)
        print(f'加载了 {len(analyzer.data)} 首歌曲的数据')
        
        # 运行各种分析
        print("\n1. 词频分析")
        word_freq = analyzer.word_frequency_analysis(top_k=20)
        print("前20个高频词:", word_freq['all_words'][:20])
        
        print("\n2. 歌词长度分析")
        length_df = analyzer.lyrics_length_analysis()
        print(f"平均词数: {length_df['word_count'].mean():.2f}")
        
        print("\n3. 词汇丰富度分析")
        ttr, types, tokens = analyzer.vocabulary_richness_analysis()
        print(f"型例比: {ttr:.4f}, 词汇类型数: {types}, 词汇标记数: {tokens}")
        
        print("\n4. 情感分析")
        sentiment_data = analyzer.sentiment_analysis()
        avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_data])
        print(f"平均情感值: {avg_sentiment:.4f}")
        
        print("\n5. LDA主题模型")
        topics, song_topics, lda_model = analyzer.lda_topic_modeling(n_topics=5)
        if topics:
            for topic in topics:
                print(f"主题 {topic['topic_id']}: {topic['top_words'][:10]}")
        
        print("\n6. 共现网络分析")
        co_occurrence_result = analyzer.co_occurrence_network_analysis(top_k=50)
        if co_occurrence_result:
            G = co_occurrence_result['graph']
        
        print("\n7. 风格演进分析")
        df, yearly_stats = analyzer.style_evolution_analysis()
        print("按年份的情感、词汇丰富度、词数统计:")
        print(yearly_stats.head())
        
        print("\n8. 新词涌现分析")
        new_words = analyzer.new_word_emergence_analysis()
        for period, words in new_words.items():
            print(f"{period} 特有词: {[w[0] for w in words[:5]]}")
    else:
        print(f'预处理数据文件不存在: {data_path}')
        print('请先运行预处理模块生成 preprocessed_lyrics.json 文件')
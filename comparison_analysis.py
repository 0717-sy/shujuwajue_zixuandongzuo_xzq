import jieba
import json
import re
import os
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from snownlp import SnowNLP
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class ComparisonAnalyzer:
    def __init__(self):
        self.xzq_words = [
            '丑八怪', '演员', '绅士', '方圆几里', '配合', '分开时',
            '天外来物', '认真的雪', '意外', '刚好', '刚刚好', '动物世界',
            '租购', '天份', '那是什么', '离开北京', '陪你去流浪', '小尖尖',
            '暧昧', '你还要我怎样', '耗尽', '一半', '怪咖', '我好像在哪见过你',
            '像风一样', '我终于成了别人的女人', '我知道你都知道', '爱我的人',
            '谢谢你', '黄色枫叶', '爱的期限', '给我的爱人', '我们的世界', '倾城',
            '红尘女子', '你过得好吗', '马戏小丑', '传说', '我的雅典娜', '未完成的歌',
            '梦开始的原点', '深深爱过你', '星河之役', '流星的眼泪', '续雪', '丢手绢',
            '朋友你们还好吗', '爱情宣判', '苏黎世的从前', '我的Show', '快乐帮', '爱不走',
            '钗头凤', '王子归来', 'OK OK', '其实', '演员', '绅士', '配合', '分开时',
            '天外来物', '认真的雪', '意外', '刚好', '刚刚好', '动物世界', '租购', '天份',
        ]
        
        # 添加自定义词典
        for word in self.xzq_words:
            jieba.add_word(word)
        
        # 定义停用词
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '它', '他', '她', '们', '这个', '那个', '这些', '那些',
            '可以', '这个', '那个', '什么', '怎么', '为什么', '如何', '这里', '那里',
            '然后', '但是', '或者', '如果', '虽然', '因为', '所以', '但是', '然后',
            '啊', '哦', '嗯', '呀', '哇', '嘿', '呃', '呀', '呢', '吧', '嘛', '啊',
            '啦', '哈', '呵', '哼', '哼', '哼', '哼', '哼', '哼', '哼', '哼', '哼',
            '他', '她', '它', '其', '该', '各', '每', '某些', '一切', '所有', '全部',
            '都', '全部', '全部', '全部', '全部', '全部', '全部', '全部', '全部', '全部',
            '这', '那', '哪', '这', '那', '哪', '这', '那', '哪', '这', '那', '哪',
            '和', '与', '及', '以及', '或', '或者', '还是', '还是', '还是', '还是', '还是', '还是',
            '为', '为了', '因为', '由于', '以便', '所以', '因此', '于是', '然后', '接着',
            '在', '于', '到', '从', '由', '被', '把', '对', '对于', '关于', '至于',
            '与', '和', '及', '以及', '或', '或者', '还是', '还是', '还是', '还是', '还是', '还是',
            '将', '把', '让', '使', '叫', '给', '为', '替', '向', '朝', '往', '对',
            '如果', '要是', '假如', '倘若', '万一', '要是', '假使', '若是', '设若', '假若',
            '虽然', '尽管', '固然', '纵然', '即使', '纵使', '哪怕', '就算', '即使',
            '因为', '由于', '鉴于', '因', '缘于', '起因', '由于', '因为', '由于', '由于',
            '因此', '所以', '因而', '故', '故而', '于是', '于是乎', '所以', '因此', '因此',
            '然而', '但是', '但', '不过', '只是', '然而', '可是', '可是', '可是', '可是', '可是', '可是',
            '而且', '并且', '以及', '还有', '此外', '另外', '加之', '又', '也', '还',
            '例如', '比如', '像', '如', '好像', '似乎', '仿佛', '犹如', '如同', '好比',
            '关于', '至于', '对于', '就', '对', '跟', '同', '与', '和', '及', '以及',
            '时', '时候', '当', '在', '于', '到', '从', '由', '自', '自从', '打',
            '能', '能够', '可以', '会', '可能', '或许', '也许', '大概', '也许', '可能',
            '将', '要', '就要', '快要', '快要', '快要', '快要', '快要', '快要', '快要',
            '已', '已经', '曾', '曾经', '早已', '已经', '曾经', '曾经', '曾经', '曾经', '曾经', '曾经',
            '未', '没有', '没', '尚未', '未', '没有', '没', '没', '没', '没', '没', '没',
            '很', '非常', '特别', '十分', '极其', '格外', '挺', '挺', '挺', '挺', '挺', '挺',
            '都', '全', '全部', '全都', '统统', '所有', '所有', '所有', '所有', '所有', '所有', '所有',
            '只', '仅仅', '只是', '不过', '光', '仅', '只', '仅仅', '只是', '不过', '光', '仅',
            '也', '又', '还', '亦', '也', '又', '还', '亦', '也', '又', '还', '亦',
            '就', '便', '于是', '于是', '于是', '于是', '于是', '于是', '于是', '于是', '于是', '于是',
            '才', '方', '才', '才', '才', '才', '才', '才', '才', '才', '才', '才',
            '却', '但', '然而', '可是', '只是', '却', '但', '然而', '可是', '只是', '却', '但',
            '等', '等等', '之类', '等', '等等', '等等', '等等', '等等', '等等', '等等', '等等', '等等'
        ])
    
    def load_jaychou_lyrics(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 周杰伦知名歌曲列表
        jaychou_songs = [
            '龙卷风', '双截棍', '安静', '简单爱', '忍者', '开不了口', '半岛铁盒', '上海一九四三', '对不起', '她的睫毛', '反方向的钟', '龙卷风',
            '爱在西元前', '爸我回来了', '说好的幸福呢', '最长的电影', '牛仔很忙', '彩虹', '青花瓷', '听妈妈的话', '千里之外',
            '本草纲目', '菊花台', '夜曲', '发如雪', '东风破', '晴天', '七里香', '半岛铁盒', '暗号', '回到过去', '轨迹', '断了的弦',
            '将军', '搁浅', '世界末日', '米兰小铁匠', '分裂', '爷爷', '娘子', '斗牛', '黑色幽默', '伊斯坦堡', '印地安老斑鸠',
            '爷爷泡的茶', '半兽人', '米兰小铁匠', '分裂', '爸我回来了', '白色风车', '稻香', '说好的幸福呢', '给我一首歌的时间',
            '烟花易冷', '雨下一整晚', '嘻哈空姐', '阳光宅男', '夜的第七章', '牛仔很忙', '天使的指纹', '乔克叔叔', '免费教学录影带',
            '逆鳞', '麦芽糖', '乱舞春秋', '飘移', '一路向北', '红模仿', '黑色毛衣', '珊瑚海', '四面楚歌', '我的地盘', '心雨',
            '枫', '浪漫手机', '退后', '白色恋人', '迷迭香', '珊瑚海', '三年二班', '东风破', '双刀', '龙拳', '火车叨位去',
            '分裂', '爷爷', '回到过去', '安静', '星晴', '简单爱', '忍者', '开不了口', '同一种调调', '印地安老斑鸠', '龙卷风',
            '反方向的钟', '威廉古堡', '双截棍', '爱在西元前', '半岛铁盒', '半岛铁盒', '安静', '轨迹', '她的睫毛', '借口',
            '将军', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅', '搁浅'
        ]
        
        # 按歌曲分割歌词（根据歌曲名进行分割）
        lines = content.split('\n')
        songs = []
        current_song = None
        current_lyrics = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是歌曲名
            if line in jaychou_songs or '周杰伦' in line or line in jaychou_songs:
                if current_song and current_lyrics:
                    songs.append({
                        'song_name': current_song,
                        'lyric': '\n'.join(current_lyrics)
                    })
                
                current_song = line.replace('周杰伦', '').strip()
                current_lyrics = []
            else:
                current_lyrics.append(line)
        
        # 添加最后一首歌
        if current_song and current_lyrics:
            songs.append({
                'song_name': current_song,
                'lyric': '\n'.join(current_lyrics)
            })
        
        if len(songs) < 10: 
            content_parts = content.split('\n\n\n') 
            songs = []
            for i, part in enumerate(content_parts):
                if part.strip():
                    songs.append({
                        'song_name': f'周杰伦歌曲_{i+1}',
                        'lyric': part.strip()
                    })

        if not songs:
            songs = [{
                'song_name': '周杰伦歌词集',
                'lyric': content
            }]
        
        return songs
    
    def preprocess_lyrics(self, lyrics_data):
        processed_data = []
        
        for song in lyrics_data:
            processed_song = song.copy()
            
            # 清理歌词
            lyric = song.get('lyric', '')
            # 移除时间戳
            cleaned_lyric = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', lyric)
            cleaned_lyric = re.sub(r'\[\d{2}:\d{2}\]', '', cleaned_lyric)
            
            # 移除标记行
            lines = cleaned_lyric.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 跳过标记行
                if re.match(r'^\[.*\]$', line) or any(marker in line for marker in 
                    ['[Intro]', '[Verse]', '[Chorus]', '[Bridge]', '[Interlude]', 
                     '[Outro]', '[Pre-Chorus]', '[Hook]', '[Solo]', '[Break]', 
                     '[Refrain]', '[Post-Chorus]']):
                    continue
                # 跳过制作信息行
                if any(keyword in line for keyword in ['作词', '作曲', '编曲', '制作', '演唱']):
                    continue
                cleaned_lines.append(line)
            
            cleaned_lyric = '\n'.join(cleaned_lines)
            processed_song['cleaned_lyric'] = cleaned_lyric
            
            # 分词
            words = jieba.lcut(cleaned_lyric)
            words = [word for word in words if word not in self.stopwords and word.strip() != '' and len(word) > 1]
            processed_song['words'] = words
            
            # 按词性分词
            from jieba import posseg as pseg
            words_with_pos = list(pseg.cut(cleaned_lyric))
            noun_words = [word for word, flag in words_with_pos if flag.startswith('n') and word not in self.stopwords and len(word) > 1]
            verb_words = [word for word, flag in words_with_pos if flag.startswith('v') and word not in self.stopwords and len(word) > 1]
            adj_words = [word for word, flag in words_with_pos if (flag.startswith('a') or flag.startswith('ad') or flag.startswith('an')) and word not in self.stopwords and len(word) > 1]
            
            processed_song['nouns'] = noun_words
            processed_song['verbs'] = verb_words
            processed_song['adjectives'] = adj_words
            
            processed_data.append(processed_song)
        
        return processed_data
    
    def unique_word_analysis(self, xzq_data, jaychou_data):
        # 获取薛之谦的词频
        xzq_words = []
        for song in xzq_data:
            xzq_words.extend(song.get('words', []))
        xzq_word_freq = Counter(xzq_words)
        
        # 获取周杰伦的词频
        jaychou_words = []
        for song in jaychou_data:
            jaychou_words.extend(song.get('words', []))
        jaychou_word_freq = Counter(jaychou_words)
        
        # 计算TF-IDF找出独特词
        all_words = set(xzq_word_freq.keys()) | set(jaychou_word_freq.keys())
        xzq_unique = {}
        jaychou_unique = {}
        
        for word in all_words:
            xzq_tf = xzq_word_freq.get(word, 0)
            jaychou_tf = jaychou_word_freq.get(word, 0)
            
            if xzq_tf > jaychou_tf and xzq_tf > 2: 
                xzq_unique[word] = xzq_tf
            elif jaychou_tf > xzq_tf and jaychou_tf > 2: 
                jaychou_unique[word] = jaychou_tf
        
        # 确保返回的字典键是字符串类型
        xzq_unique_str = {str(k): v for k, v in xzq_unique.items()}
        jaychou_unique_str = {str(k): v for k, v in jaychou_unique.items()}
        
        return {
            'xzq_unique': sorted(xzq_unique_str.items(), key=lambda x: x[1], reverse=True)[:50],
            'jaychou_unique': sorted(jaychou_unique_str.items(), key=lambda x: x[1], reverse=True)[:50]
        }
    
    def get_sentiment_score(self, text):
        if not text.strip():
            return 0.3 
        try:
            s = SnowNLP(text)
            sentiment_score = s.sentiments
            lyric_lower = text.lower()
            negative_keywords = ['不', '没', '无', '恨', '痛', '伤', '泪', '哭', '错', '悔', '恨', '怨', '怒', '悲', '愁', '哀', '苦', '痛', '难', '烦', '厌', '惧', '怕', '绝望', '放弃', '痛恨', '讨厌', '嫌弃', '难过', '伤心', '失望', '沮丧', '孤独', '寂寞', '痛苦', '折磨', '分手', '离别', '背叛', '辜负', '心碎', '眼泪', '眼泪', '眼泪', '眼泪', '眼泪']
            negative_count = sum(1 for keyword in negative_keywords if keyword in lyric_lower)
            positive_keywords = ['爱', '喜', '笑', '乐', '好', '美', '甜', '暖', '幸', '福', '欢', '悦', '赞', '好', '妙', '棒', '赞', '美', '爱', '恋', '甜', '心', '情', '喜欢', '开心', '快乐', '幸福', '温暖', '感动', '美好', '希望', '阳光', '微笑', '拥抱', '期待', '感谢', '感恩', '珍惜', '美好', '美好', '美好', '美好', '美好']
            positive_count = sum(1 for keyword in positive_keywords if keyword in lyric_lower)
            total_keywords = positive_count + negative_count
            if total_keywords > 0:
                sentiment_adjustment = (positive_count - negative_count) / total_keywords
                sentiment_score = max(0.1, min(0.9, sentiment_score + sentiment_adjustment * 0.4))
                if negative_count > positive_count:
                    sentiment_score = sentiment_score * 0.8 
            else:
                sentiment_score = sentiment_score * 0.9
            sentiment_score = max(0.1, min(0.9, sentiment_score))
            
            return sentiment_score
        except:
            return 0.3  
    
    def visualize_comparison(self, comparison_result, output_path='utils/visualization/comparison_analysis.png'):
        plt.figure(figsize=(16, 12))
        
        # 绘制薛之谦独特词
        plt.subplot(2, 2, 1)
        if comparison_result['xzq_unique']:
            xzq_words, xzq_freqs = zip(*comparison_result['xzq_unique'][:15])
            plt.barh(range(len(xzq_words)), xzq_freqs, color='#FF6B6B')
            plt.yticks(range(len(xzq_words)), xzq_words)
            plt.xlabel('词频')
            plt.title('薛之谦独特高频词')
            plt.gca().invert_yaxis()
        
        # 绘制周杰伦独特词
        plt.subplot(2, 2, 2)
        if comparison_result['jaychou_unique']:
            jaychou_words, jaychou_freqs = zip(*comparison_result['jaychou_unique'][:15])
            plt.barh(range(len(jaychou_words)), jaychou_freqs, color='#4ECDC4')
            plt.yticks(range(len(jaychou_words)), jaychou_words)
            plt.xlabel('词频')
            plt.title('周杰伦独特高频词')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'对比分析图已保存到: {output_path}')
    
    def run_full_comparison(self, xzq_file, jaychou_file):
        
        # 加载薛之谦数据
        with open(xzq_file, 'r', encoding='utf-8') as f:
            xzq_data = json.load(f)
        
        # 预处理薛之谦数据
        xzq_processed = self.preprocess_lyrics(xzq_data)
        print(f'薛之谦数据预处理完成，共{xzq_processed.__len__()}首歌曲')
        
        # 加载周杰伦数据
        jaychou_raw = self.load_jaychou_lyrics(jaychou_file)
        jaychou_processed = self.preprocess_lyrics(jaychou_raw)
        print(f'周杰伦数据预处理完成，共{jaychou_processed.__len__()}首歌曲')
        
        # 独特词分析
        unique_analysis = self.unique_word_analysis(xzq_processed, jaychou_processed)
        
        # 可视化
        self.visualize_comparison(unique_analysis)
        
        # 输出结果
        print('\n=== 薛之谦独特高频词 Top 10 ===')
        for i, (word, freq) in enumerate(unique_analysis['xzq_unique'][:10], 1):
            print(f'{i:2d}. {word}: {freq}次')
        
        print('\n=== 周杰伦独特高频词 Top 10 ===')
        for i, (word, freq) in enumerate(unique_analysis['jaychou_unique'][:10], 1):
            print(f'{i:2d}. {word}: {freq}次')
        
        return {
            'unique_analysis': unique_analysis
        }

if __name__ == '__main__':
    analyzer = ComparisonAnalyzer()
    
    # 运行对比分析
    result = analyzer.run_full_comparison(
        'utils/data/xuezhiqian_lyrics_valid_final.json',
        'utils/jaychou_lyrics.txt'
    )
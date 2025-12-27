import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import os
import networkx as nx
from matplotlib.font_manager import FontProperties
from collections import Counter
import jieba

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class LyricVisualizer:
    def __init__(self):
        # 检查中文字体
        self.font_path = self.find_simsun_font()
    
    def find_simsun_font(self):
        """查找系统中的宋体或黑体字体"""
        # 常见的中文字体路径
        common_font_paths = [
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
            r'C:\Windows\Fonts\msyh.ttf',   # 微软雅黑
            r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux 文泉驿微米黑
        ]
        
        for font_path in common_font_paths:
            if os.path.exists(font_path):
                return font_path
        
        # 如果找不到，返回None，wordcloud会使用默认字体
        return None
    
    def generate_wordcloud(self, word_counts, output_path='utils/visualization/wordcloud.png'):
        """生成圆形词云图 - 科技感配色"""
        if not word_counts:
            return None
        
        # 创建圆形遮罩
        x, y = np.ogrid[:800, :800]
        mask = (x - 400) ** 2 + (y - 400) ** 2 > 380 ** 2
        mask = 255 * mask.astype(int)
        
        # 创建词云对象 - 使用更美观的配色
        wc = WordCloud(
            font_path=self.font_path,
            background_color='white', 
            colormap='viridis',  # 使用更现代、渐变的色图
            max_words=200,  # 增加词汇数量
            width=1000,
            height=1000,
            margin=2,
            mask=mask,
            relative_scaling=0.5,
            min_font_size=8,  # 减小最小字体以显示更多词汇
            max_font_size=100,
            font_step=2
        )
        
        # 生成词云
        wc.generate_from_frequencies(word_counts)
        
        # 显示词云
        plt.figure(figsize=(10, 7), facecolor='black')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('薛之谦歌词高频词汇云图', fontsize=16, fontweight='bold', color='white')
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        # 关闭图形以释放内存
        plt.close()
        
        return wc
    
    def generate_pos_wordcloud(self, pos_words_dict, output_path='utils/visualization/pos_wordcloud.png'):
        """生成按词性分类的词云图"""
        if not pos_words_dict:
            return None
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('薛之谦歌词词性词云图', fontsize=18, fontweight='bold')
        
        pos_names = ['名词', '动词', '形容词', '否定词']
        pos_keys = ['nouns', 'verbs', 'adjectives', 'negations']
        
        for idx, (pos_name, pos_key) in enumerate(zip(pos_names, pos_keys)):
            ax = axes[idx//2, idx%2]
            
            if pos_key in pos_words_dict and pos_words_dict[pos_key]:
                # 统计词频
                word_freq = Counter()
                for word, freq in pos_words_dict[pos_key][:30]:  # 只取前30个高频词
                    word_freq[word] = freq
                
                if word_freq:
                    # 生成词云
                    wc = WordCloud(
                        font_path=self.font_path,
                        background_color='white',
                        max_words=30,
                        width=400,
                        height=400,
                        margin=2
                    )
                    
                    wc.generate_from_frequencies(word_freq)
                    
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'{pos_name}词云', fontsize=14, fontweight='bold')
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.close()
        
        return fig
    
    def plot_top_words(self, top_words, output_path='utils/visualization/top_words.png'):
        """绘制高频词汇条形图"""
        if not top_words:
            return None
        
        # 提取词汇和频率
        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]
        
        # 创建条形图 - 学术风格改进
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(words)), counts, color='#4285F4', alpha=0.85)  # 更专业的颜色
        plt.yticks(range(len(words)), words, fontsize=13, fontweight='medium')
        plt.xlabel('出现频率', fontsize=15, fontweight='bold', labelpad=12)
        plt.ylabel('词汇', fontsize=15, fontweight='bold', labelpad=12)
        plt.title('薛之谦歌词高频词汇排行榜', fontsize=18, fontweight='bold', pad=15)
        plt.gca().invert_yaxis()  # 使最高频的词汇在顶部
        
        # 添加网格线
        plt.grid(axis='x', linestyle='--', alpha=0.7, linewidth=0.8)
        
        # 优化坐标轴
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        # 为每个条形添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{counts[i]}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_sentiment_trend(self, sentiment_results, output_path='utils/visualization/sentiment_trend.png'):
        """绘制情感趋势图，按时间分时期展示，柱状图加折线图形式"""
        if not sentiment_results:
            return None
        
        # 按时期分组数据
        period_data = {}
        for result in sentiment_results:
            period = result.get('period', 'Unknown')
            if period not in period_data:
                period_data[period] = []
            period_data[period].append(result)
        
        # 为不同情感标签分配颜色
        color_map = {'积极': '#27AE60', '中性': '#3498DB', '消极': '#E74C3C'}
        
        # 按时期绘制情感分析图
        for period_idx, (period, period_songs) in enumerate(period_data.items()):
            # 准备当前时期的数据
            song_names = [result['song_name'] for result in period_songs]
            scores = [result['sentiment_score'] for result in period_songs]
            
            # 检查是否有情感标签，如果没有则根据情感值自动生成
            if 'sentiment_label' in period_songs[0]:
                labels = [result['sentiment_label'] for result in period_songs]
            else:
                # 根据情感值自动分类
                labels = []
                for score in scores:
                    if score < 0.4:
                        labels.append('消极')
                    elif score > 0.6:
                        labels.append('积极')
                    else:
                        labels.append('中性')
            
            bar_colors = [color_map.get(label, '#95a5a6') for label in labels]
            
            # 动态调整图表尺寸，确保歌曲名称显示清晰
            max_song_name_length = max(len(name) for name in song_names) if song_names else 10
            fig_width = max(16, min(len(song_names) * 0.8, 30))  # 限制最大宽度
            fig_height = max(8, min(len(song_names) * 0.3, 15))  # 根据歌曲数量调整高度
            
            plt.figure(figsize=(fig_width, fig_height), facecolor='white')
            
            # 创建柱状图和折线图的组合图
            fig, ax1 = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
            
            # 绘制柱状图
            y_pos = np.arange(len(song_names))
            bars = ax1.barh(y_pos, scores, color=bar_colors, height=0.6, alpha=0.7)
            
            # 设置y轴标签 - 使用水平方向避免重叠
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(song_names, fontsize=9)
            
            # 绘制折线图
            ax1.plot(scores, y_pos, color='red', marker='o', linewidth=2, markersize=6, label='情感趋势线')
            
            # 添加情感分类标签和得分
            for i, (score, label) in enumerate(zip(scores, labels)):
                # 情感得分
                ax1.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=8, weight='bold')
                # 情感标签
                ax1.text(0.85, i, f'({label})', va='center', fontsize=8, weight='bold', 
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # 设置坐标轴标签和标题
            ax1.set_xlabel('情感值 (0=消极, 1=积极)', fontsize=12, labelpad=10)
            ax1.set_ylabel('歌曲名称', fontsize=12, labelpad=10)
            ax1.set_title(f'薛之谦歌词情感分析 - {period}', fontsize=14, pad=20)
            
            # 设置x轴范围
            ax1.set_xlim(0, 1.1)
            
            # 添加网格线
            ax1.grid(axis='x', linestyle='--', alpha=0.7, linewidth=0.8)
            
            # 设置图表背景
            ax1.set_facecolor('white')
            
            # 调整布局，确保标签不被截断
            plt.tight_layout()
            
            # 保存图片
            if output_path:
                # 为每个时期生成不同的文件名
                output_path_split = output_path.rsplit('.', 1)
                period_output_path = f"{output_path_split[0]}_{period}.{output_path_split[1]}".replace(' ', '_').replace('(', '').replace(')', '')
                plt.savefig(period_output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"情感分析图已生成")
            
            plt.close()
    
    def plot_topic_distribution(self, topics, output_path='utils/visualization/topic_distribution.png'):
        """绘制主题分布饼图"""
        if not topics:
            return None
        
        # 准备数据
        topic_ids = [topic['topic_id'] for topic in topics]
        # 简化主题名称，更适合学术展示
        topic_names = [f"主题{topic['topic_id']}" for topic in topics]
        
        # 创建饼图 - 学术风格改进
        plt.figure(figsize=(12, 10))
        
        # 使用更专业的颜色方案
        colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#9C27B0', '#FF6D00', '#00ACC1']
        
        wedges, texts, autotexts = plt.pie(
            [1] * len(topics),  # 简单分布，实际数据可以根据每个主题的文档数调整
            labels=topic_names,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
            textprops={'fontsize': 13, 'fontweight': 'medium'}
        )
        
        # 设置百分比标签的样式
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        plt.title('薛之谦歌词主题分布', fontsize=18, fontweight='bold', pad=20)
        plt.axis('equal')  # 保证饼图为圆形
        
        # 添加图例，显示详细主题信息
        legend_labels = [f"主题{topic['topic_id']}: {', '.join(topic['top_words'][:3])}" for topic in topics]
        plt.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_lexical_features(self, lexical_feature_analysis, output_dir=None):
        """可视化词汇特征分析结果"""
        if not lexical_feature_analysis or 'word_frequency' not in lexical_feature_analysis:
            return None
        
        word_frequency = lexical_feature_analysis['word_frequency']
        
        # 可视化名词高频词
        if 'nouns' in word_frequency and word_frequency['nouns']:
            output_path = os.path.join(output_dir, "名词高频词.png") if output_dir else None
            self.plot_word_frequency_by_pos(word_frequency['nouns'], "名词", output_path)
        
        # 可视化动词高频词
        if 'verbs' in word_frequency and word_frequency['verbs']:
            output_path = os.path.join(output_dir, "动词高频词.png") if output_dir else None
            self.plot_word_frequency_by_pos(word_frequency['verbs'], "动词", output_path)
        
        # 可视化形容词高频词
        if 'adjectives' in word_frequency and word_frequency['adjectives']:
            output_path = os.path.join(output_dir, "形容词高频词.png") if output_dir else None
            self.plot_word_frequency_by_pos(word_frequency['adjectives'], "形容词", output_path)
        
        # 可视化歌词长度分析
        if 'lyric_lengths' in lexical_feature_analysis:
            output_path = os.path.join(output_dir, "歌词长度分布.png") if output_dir else None
            self.plot_lyric_length_distribution(lexical_feature_analysis['lyric_lengths'], output_path)
            
            output_path = os.path.join(output_dir, "歌词长度按时期分布.png") if output_dir else None
            self.plot_lyric_length_by_period(lexical_feature_analysis['lyric_lengths'], output_path)
            
            output_path = os.path.join(output_dir, "歌词长度统计.png") if output_dir else None
            self.plot_lyric_length_stats(lexical_feature_analysis['lyric_lengths'], output_path)
            
    def plot_lyric_length_distribution(self, lyric_lengths, output_path='utils/visualization/lyric_length_distribution.png'):
        """绘制歌词长度分布直方图"""
        if not lyric_lengths or not lyric_lengths['details']:
            return None
        
        # 准备数据
        char_counts = [item['char_count'] for item in lyric_lengths['details']]
        word_counts = [item['word_count'] for item in lyric_lengths['details']]
        
        # 创建直方图
        plt.figure(figsize=(15, 6))
        
        # 字符数分布
        plt.subplot(1, 2, 1)
        plt.hist(char_counts, bins=20, color='#4A90E2', alpha=0.7)
        plt.xlabel('歌词字符数', fontsize=14)
        plt.ylabel('歌曲数量', fontsize=14)
        plt.title('薛之谦歌词字符数分布', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # 词数分布
        plt.subplot(1, 2, 2)
        plt.hist(word_counts, bins=20, color='#50E3C2', alpha=0.7)
        plt.xlabel('歌词词数', fontsize=14)
        plt.ylabel('歌曲数量', fontsize=14)
        plt.title('薛之谦歌词词数分布', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"歌词长度分布图已保存到: {output_path}")
        
        plt.close()
        
    def plot_lyric_length_by_period(self, lyric_lengths, output_path='utils/visualization/lyric_length_by_period.png'):
        """按时间周期绘制歌词长度对比图"""
        if not lyric_lengths or not lyric_lengths['details']:
            return None
        
        # 按时期分组
        periods = {'早期 (2006-2012)': {'char_counts': [], 'word_counts': []}, 
                   '中期 (2013-2017)': {'char_counts': [], 'word_counts': []}, 
                   '近期 (2018至今)': {'char_counts': [], 'word_counts': []}}
        
        for item in lyric_lengths['details']:
            period = item.get('period', '中期 (2013-2017)')  # 默认中期
            
            # 处理不同格式的period值
            if period in ['2006-2012', '早期 (2006-2012)']:
                period = '早期 (2006-2012)'
            elif period in ['2013-2017', '2015-2017', '中期 (2013-2017)']:
                period = '中期 (2013-2017)'
            elif period in ['2018至今', '2018-至今', '近期 (2018至今)']:
                period = '近期 (2018至今)'
            
            if period in periods:
                periods[period]['char_counts'].append(item['char_count'])
                periods[period]['word_counts'].append(item['word_count'])
        
        # 准备数据
        period_names = list(periods.keys())
        avg_char_counts = [sum(data['char_counts']) / len(data['char_counts']) if data['char_counts'] else 0 for data in periods.values()]
        avg_word_counts = [sum(data['word_counts']) / len(data['word_counts']) if data['word_counts'] else 0 for data in periods.values()]
        
        # 创建对比图
        plt.figure(figsize=(12, 8))
        
        # 设置柱状图的位置
        x = np.arange(len(period_names))
        width = 0.35
        
        # 绘制字符数对比
        bars1 = plt.bar(x - width/2, avg_char_counts, width, label='平均字符数', color='#3498db')
        # 绘制词数对比
        bars2 = plt.bar(x + width/2, avg_word_counts, width, label='平均词数', color='#e74c3c')
        
        # 添加标签和标题
        plt.xlabel('创作时期', fontsize=14)
        plt.ylabel('数量', fontsize=14)
        plt.title('薛之谦不同时期歌词长度对比', fontsize=16)
        plt.xticks(x, period_names)
        plt.legend()
        
        # 在柱状图上添加数值标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.0f}', ha='center', va='bottom')
        
        add_labels(bars1)
        add_labels(bars2)
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"歌词长度按时期对比图已保存到: {output_path}")
        
        plt.close()
        
    def plot_lyric_length_stats(self, lyric_lengths, output_path='utils/visualization/lyric_length_stats.png'):
        """绘制歌词长度统计信息"""
        if not lyric_lengths:
            return None
        
        # 准备数据
        details = lyric_lengths['details']
        char_counts = [item['char_count'] for item in details]
        word_counts = [item['word_count'] for item in details]
        
        # 计算统计信息
        stats = {
            '平均字符数': lyric_lengths.get('avg_char_count', sum(char_counts) / len(char_counts)),
            '平均词数': lyric_lengths.get('avg_word_count', sum(word_counts) / len(word_counts)),
            '最长字符数': max(char_counts),
            '最短字符数': min(char_counts),
            '最长词数': max(word_counts),
            '最短词数': min(word_counts)
        }
        
        # 创建统计信息图表
        plt.figure(figsize=(12, 8))
        
        # 提取标签和值
        labels = list(stats.keys())
        values = list(stats.values())
        
        # 创建水平条形图
        plt.barh(range(len(labels)), values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#8e44ad'])
        plt.yticks(range(len(labels)), labels, fontsize=12)
        plt.xlabel('数值', fontsize=14)
        plt.ylabel('统计指标', fontsize=14)
        plt.title('薛之谦歌词长度统计信息', fontsize=16)
        
        # 在条形图上添加数值标签
        for i, value in enumerate(values):
            plt.text(value + 5, i, f'{value:.0f}', va='center', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"歌词长度统计信息图已保存到: {output_path}")
        
        plt.close()
        
    def plot_lyric_length_trend(self, lyric_lengths, output_path='utils/visualization/lyric_length_trend.png'):
        """绘制歌词长度随时间的变化趋势"""
        if not lyric_lengths or 'trend_analysis' not in lyric_lengths:
            return None
        
        trend_analysis = lyric_lengths['trend_analysis']
        period_char_avg = trend_analysis.get('period_char_avg', {})
        period_word_avg = trend_analysis.get('period_word_avg', {})
        
        # 准备数据 - 按时间顺序排序
        period_order = ['早期 (2006-2012)', '中期 (2013-2017)', '近期 (2018至今)']
        periods = [p for p in period_order if p in period_char_avg]
        avg_char = [period_char_avg[p] for p in periods]
        avg_word = [period_word_avg[p] for p in periods]
        
        # 创建趋势图
        plt.figure(figsize=(12, 8))
        
        # 设置x轴位置
        x = np.arange(len(periods))
        width = 0.35
        
        # 绘制字符数趋势
        bars1 = plt.bar(x - width/2, avg_char, width, label='平均字符数', color='#3498db')
        # 绘制词数趋势
        bars2 = plt.bar(x + width/2, avg_word, width, label='平均词数', color='#e74c3c')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.0f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.xlabel('时期', fontsize=14)
        plt.ylabel('平均值', fontsize=14)
        plt.title('薛之谦歌词长度随时间的变化趋势', fontsize=16)
        plt.xticks(x, periods, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"歌词长度趋势图已保存到: {output_path}")
            
    def plot_unique_words_comparison(self, unique_words_analysis, output_path=None):
        """绘制薛之谦与周杰伦歌词的独特高频词对比图"""
        if not unique_words_analysis:
            return None
            
        # 准备数据
        xue_unique = unique_words_analysis.get('xuezhiqian_unique_words', [])[:15]  # 只取前15个独特词
        jay_unique = unique_words_analysis.get('jaychou_unique_words', [])[:15]  # 只取前15个独特词
        
        # 提取词汇和TF-IDF分数
        xue_words = [word for word, score in xue_unique]
        xue_scores = [score for word, score in xue_unique]
        jay_words = [word for word, score in jay_unique]
        jay_scores = [score for word, score in jay_unique]
        
        # 创建对比图
        plt.figure(figsize=(15, 12))
        
        # 设置布局
        plt.subplot(2, 1, 1)
        
        # 绘制薛之谦的独特词
        x1 = np.arange(len(xue_words))
        bars1 = plt.barh(x1, xue_scores, color='#3498db')
        plt.yticks(x1, xue_words, fontsize=12)
        plt.xlabel('TF-IDF分数', fontsize=14)
        plt.title('薛之谦歌词独特高频词', fontsize=16)
        plt.gca().invert_yaxis()  # 使最高频的词在顶部
        
        # 添加数值标签
        for i, score in enumerate(xue_scores):
            plt.text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=10)
        
        plt.subplot(2, 1, 2)
        
        # 绘制周杰伦的独特词
        x2 = np.arange(len(jay_words))
        bars2 = plt.barh(x2, jay_scores, color='#e74c3c')
        plt.yticks(x2, jay_words, fontsize=12)
        plt.xlabel('TF-IDF分数', fontsize=14)
        plt.title('周杰伦歌词独特高频词', fontsize=16)
        plt.gca().invert_yaxis()  # 使最高频的词在顶部
        
        # 添加数值标签
        for i, score in enumerate(jay_scores):
            plt.text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=10)
        
        plt.tight_layout(h_pad=5)
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"独特词对比图已保存到: {output_path}")
        
        plt.close()
        
    def plot_type_token_ratio(self, vocabulary_richness, output_path=None):
        """绘制型例比分析图"""
        if not vocabulary_richness:
            return None
        
        # 1. 绘制各时期平均型例比趋势
        period_ttr_avg = vocabulary_richness.get('period_ttr_avg', {})
        
        # 按时间顺序排序
        period_order = ['早期 (2006-2012)', '中期 (2013-2017)', '近期 (2018至今)']
        periods = [p for p in period_order if p in period_ttr_avg]
        ttr_values = [period_ttr_avg[p] for p in periods]
        
        plt.figure(figsize=(15, 6))
        
        # 绘制趋势线
        plt.subplot(1, 2, 1)
        plt.plot(periods, ttr_values, marker='o', linewidth=2, markersize=8, color='#3498db')
        plt.xlabel('时期', fontsize=14)
        plt.ylabel('型例比 (TTR)', fontsize=14)
        plt.title('薛之谦歌词型例比随时间的变化趋势', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, value in enumerate(ttr_values):
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center', fontsize=12)
        
        # 2. 绘制每首歌的型例比分布
        plt.subplot(1, 2, 2)
        song_ttr = vocabulary_richness.get('song_type_token_ratios', [])
        if song_ttr:
            ttr_values_song = [item['type_token_ratio'] for item in song_ttr]
            plt.hist(ttr_values_song, bins=20, color='#e74c3c', alpha=0.7)
            plt.xlabel('型例比 (TTR)', fontsize=14)
            plt.ylabel('歌曲数量', fontsize=14)
            plt.title('薛之谦歌词型例比分布', fontsize=16)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"型例比分析图已保存到: {output_path}")
        
        plt.close()
    
    def plot_word_frequency_by_pos(self, word_counts, pos_type, output_path=None):
        """按词性绘制词频条形图"""
        if not word_counts:
            return None
        
        # 提取词汇和频率
        words = [word for word, count in word_counts]
        counts = [count for word, count in word_counts]
        
        # 创建条形图 - 学术风格改进
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(words)), counts, color='#4285F4', alpha=0.85)
        plt.yticks(range(len(words)), words, fontsize=13, fontweight='medium')
        plt.xlabel('出现频率', fontsize=15, fontweight='bold', labelpad=12)
        plt.ylabel(pos_type, fontsize=15, fontweight='bold', labelpad=12)
        plt.title(f'薛之谦歌词高频{pos_type}排行榜', fontsize=18, fontweight='bold', pad=15)
        plt.gca().invert_yaxis()  # 使最高频的词汇在顶部
        
        # 添加网格线
        plt.grid(axis='x', linestyle='--', alpha=0.7, linewidth=0.8)
        
        # 优化坐标轴
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        # 为每个条形添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{counts[i]}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"高频{pos_type}条形图已保存到: {output_path}")
        
        plt.close()
    
    def plot_cooccurrence_network(self, network_data, output_path=None):
        """绘制高频词共现网络"""
        import networkx as nx
        
        if not network_data or 'nodes' not in network_data or 'edges' not in network_data:
            return None
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        nodes = network_data['nodes']
        node_degrees = network_data.get('node_degrees', {})
        
        # 计算节点大小（基于度数）
        if node_degrees:
            max_degree = max(node_degrees.values())
            node_sizes = [node_degrees.get(node, 1) * 2000 / max_degree for node in nodes]
        else:
            # 如果没有度数信息，使用固定大小
            node_sizes = [1000 for _ in nodes]
        
        # 添加边
        edges = network_data['edges']
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        
        # 创建图形
        plt.figure(figsize=(20, 15))
        
        # 使用更美观的布局算法
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=node_sizes, 
                             node_color='#3498db', alpha=0.8)
        
        # 绘制边
        edges_list = [(edge['source'], edge['target']) for edge in edges]
        weights = [edge['weight'] for edge in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [weight * 3 / max_weight for weight in weights]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_list, width=edge_widths, 
                             edge_color='#888888', alpha=0.6)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', 
                              font_family='SimHei' if self.font_path else 'sans-serif')
        
        # 设置标题
        plt.title('薛之谦歌词高频词共现网络', fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"共现网络图已保存到: {output_path}")
        
        plt.close()
    
    def plot_correlation_matrix(self, correlation_data, output_path=None):
        """绘制相关性热力图"""
        import seaborn as sns
        
        # 检查输入类型，如果是字典则转换为DataFrame
        if isinstance(correlation_data, dict):
            if 'correlation_matrix' in correlation_data:
                corr_df = pd.DataFrame(correlation_data['correlation_matrix'])
            elif 'correlation' in correlation_data:
                corr_df = pd.DataFrame(correlation_data['correlation'])
            else:
                return None
        elif hasattr(correlation_data, 'columns'):  # 如果已经是DataFrame
            corr_df = correlation_data
        else:
            return None
        
        # 创建热力图
        plt.figure(figsize=(12, 10), facecolor='white')
        
        # 使用seaborn绘制热力图
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                   annot_kws={'size': 12, 'weight': 'bold'}, 
                   square=True, linewidths=.5, cbar_kws={'shrink': .8})
        
        # 设置标题和标签
        plt.title('歌词特征与流行度相关性热力图', fontsize=18, fontweight='bold', pad=20)
        plt.xticks(fontsize=12, fontweight='medium')
        plt.yticks(fontsize=12, fontweight='medium')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"相关性热力图已保存到: {output_path}")
        
        plt.close()
    
    def plot_detailed_word_frequency(self, word_freq_data, output_path=None):
        """绘制详细词频统计图（按词性分类）- 科技感配色"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('薛之谦歌词高频词分析（按词性分类）', fontsize=18, fontweight='bold', color='#2C3E50')
        
        pos_names = ['名词', '动词', '形容词', '否定词']
        pos_keys = ['nouns', 'verbs', 'adjectives', 'negations']
        
        # 科技感配色方案
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # 红、蓝、绿、橙
        
        for idx, (pos_name, pos_key) in enumerate(zip(pos_names, pos_keys)):
            ax = axes[idx//2, idx%2]
            
            if pos_key in word_freq_data:
                words, freqs = [], []
                for word, freq in word_freq_data[pos_key][:15]:  # 取前15个高频词
                    words.append(word)
                    freqs.append(freq)
                
                bars = ax.barh(range(len(words)), freqs, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=10)
                ax.set_xlabel('词频', fontsize=12)
                ax.set_title(f'{pos_name}高频词', fontsize=14, fontweight='bold', color=colors[idx])
                ax.invert_yaxis()  # 使最高频的词在顶部
                
                # 添加数值标签
                for i, (bar, freq) in enumerate(zip(bars, freqs)):
                    ax.text(freq + max(freqs)*0.01, i, str(freq), va='center', fontsize=9)
                
                # 设置网格
                ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_metaphor_analysis(self, metaphor_data, output_path=None):
        """绘制比喻词分析图 - 科技感配色"""
        if not metaphor_data:
            return None
        
        # 统计比喻词的使用频率
        metaphor_counts = Counter([item['metaphor_word'] for item in metaphor_data])
        
        if not metaphor_counts:
            return None
        
        # 创建图表
        plt.figure(figsize=(12, 8), facecolor='white')
        metaphor_words, counts = zip(*metaphor_counts.most_common(10))
        
        bars = plt.bar(range(len(metaphor_words)), counts, color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
        plt.xticks(range(len(metaphor_words)), metaphor_words)
        plt.xlabel('比喻词', fontsize=12, fontweight='bold')
        plt.ylabel('使用次数', fontsize=12, fontweight='bold')
        plt.title('薛之谦歌词比喻词使用频率', fontsize=16, fontweight='bold', color='#2C3E50')
        
        # 添加数值标签
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontweight='bold')
        
        # 添加网格
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return plt.gcf()
    
    def plot_sentence_complexity(self, complexity_data, output_path=None):
        """绘制句长与复杂度分析图"""
        if not complexity_data:
            return None
        
        # 提取平均句长数据
        avg_lengths = [item['avg_line_length'] for item in complexity_data if item['avg_line_length'] > 0]
        
        if not avg_lengths:
            return None
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制直方图
        plt.hist(avg_lengths, bins=20, color='#96CEB4', edgecolor='black', alpha=0.7)
        plt.xlabel('平均句长（字数）')
        plt.ylabel('歌曲数量')
        plt.title('薛之谦歌词平均句长分布', fontsize=16, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return plt.gcf()
    
    def plot_period_comparison(self, period_stats, output_path=None):
        """绘制时期对比分析图"""
        periods = list(period_stats.keys())
        avg_sentiments = [period_stats[p]['avg_sentiment'] for p in periods]
        song_counts = [period_stats[p]['song_count'] for p in periods]
        vocab_richness = [period_stats[p]['vocabulary_richness'] for p in periods]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle('薛之谦歌词时期对比分析', fontsize=18, fontweight='bold', color='#2C3E50')
        
        # 情感对比
        axes[0, 0].bar(periods, avg_sentiments, color=['#E74C3C', '#3498DB', '#2ECC71'], alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('各时期平均情感值对比', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0, 0].set_ylabel('平均情感值')
        axes[0, 0].set_ylim(0, 1)
        # 添加数值标签
        for i, v in enumerate(avg_sentiments):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 歌曲数量对比
        axes[0, 1].bar(periods, song_counts, color=['#F39C12', '#9B59B6', '#1ABC9C'], alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('各时期歌曲数量对比', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0, 1].set_ylabel('歌曲数量')
        # 添加数值标签
        for i, v in enumerate(song_counts):
            axes[0, 1].text(i, v + max(song_counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 词汇丰富度对比
        axes[1, 0].bar(periods, vocab_richness, color=['#34495E', '#E67E22', '#8E44AD'], alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1, 0].set_title('各时期词汇丰富度对比', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[1, 0].set_ylabel('词汇丰富度(TTR)')
        # 添加数值标签
        for i, v in enumerate(vocab_richness):
            axes[1, 0].text(i, v + max(vocab_richness)*0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 歌词长度对比
        avg_char_counts = [period_stats[p]['avg_char_count'] for p in periods]
        axes[1, 1].bar(periods, avg_char_counts, color=['#D35400', '#7F8C8D', '#27AE60'], alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1, 1].set_title('各时期平均歌词长度对比', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[1, 1].set_ylabel('平均字符数')
        # 添加数值标签
        for i, v in enumerate(avg_char_counts):
            axes[1, 1].text(i, v + max(avg_char_counts)*0.01, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_time_period_sentiment_analysis(self, sentiment_results, output_path=None):
        """按时间分时期绘制情感分析图，替换箱线图为折线图和柱状图"""
        if not sentiment_results:
            return None
        
        # 按时期分组数据
        period_data = {}
        for result in sentiment_results:
            period = result.get('period', 'Unknown')
            if period not in period_data:
                period_data[period] = []
            period_data[period].append(result)
        
        # 计算各时期的统计信息
        periods = list(period_data.keys())
        avg_sentiments = []
        sentiment_stds = []
        song_counts = []
        
        for period in periods:
            period_sentiments = [r['sentiment_score'] for r in period_data[period]]
            avg_sentiments.append(np.mean(period_sentiments) if period_sentiments else 0.5)
            sentiment_stds.append(np.std(period_sentiments) if len(period_sentiments) > 1 else 0)
            song_counts.append(len(period_sentiments))
        
        # 创建组合图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle('薛之谦歌词情感分析 - 按时期统计', fontsize=18, fontweight='bold', color='#2C3E50')
        
        # 1. 各时期平均情感值柱状图
        bars1 = axes[0, 0].bar(periods, avg_sentiments, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'], 
                               alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('各时期平均情感值', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0, 0].set_ylabel('平均情感值')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, avg_sentiments):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, 
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. 各时期情感值标准差
        bars2 = axes[0, 1].bar(periods, sentiment_stds, color=['#9B59B6', '#1ABC9C', '#D35400', '#34495E'], 
                               alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('各时期情感值标准差', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0, 1].set_ylabel('标准差')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars2, sentiment_stds):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(sentiment_stds)*0.02, 
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. 各时期歌曲数量
        bars3 = axes[1, 0].bar(periods, song_counts, color=['#F1C40F', '#E67E22', '#8E44AD', '#17A589'], 
                               alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1, 0].set_title('各时期歌曲数量', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[1, 0].set_ylabel('歌曲数量')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars3, song_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(song_counts)*0.01, 
                           str(value), ha='center', va='bottom', fontweight='bold', fontsize=10)
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 4. 情感值趋势线图
        x_pos = range(len(periods))
        axes[1, 1].plot(x_pos, avg_sentiments, marker='o', linewidth=2.5, markersize=8, 
                        color='#E74C3C', label='平均情感值')
        axes[1, 1].set_title('情感值变化趋势', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[1, 1].set_ylabel('平均情感值')
        axes[1, 1].set_xlabel('时期')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(periods, rotation=45)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for i, value in enumerate(avg_sentiments):
            axes[1, 1].text(i, value + 0.03, f'{value:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
        axes[1, 1].legend()
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_random_50_songs_sentiment(self, sentiment_results, output_path='utils/visualization/random_50_songs_sentiment.png'):
        """绘制随机50首歌情感分析图"""
        # 按情感值排序
        sorted_results = sorted(sentiment_results, key=lambda x: x['sentiment_score'], reverse=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
        fig.suptitle('随机50首薛之谦歌曲情感分析', fontsize=18, fontweight='bold', color='#2C3E50')
        
        # 左图：情感值分布直方图
        sentiments = [r['sentiment_score'] for r in sentiment_results]
        axes[0].hist(sentiments, bins=15, color='#3498DB', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('情感值', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('歌曲数量', fontsize=12, fontweight='bold')
        axes[0].set_title('情感值分布直方图', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 右图：情感值散点图（按时期着色）
        periods = [r['period'] for r in sentiment_results]
        unique_periods = list(set(periods))
        colors_map = {'2006-2012': '#E74C3C', '2013-2017': '#3498DB', '2018-至今': '#2ECC71', 'Unknown': '#95A5A6'}
        
        for period in unique_periods:
            period_data = [(i, r['sentiment_score']) for i, r in enumerate(sentiment_results) if r['period'] == period]
            if period_data:
                indices, scores = zip(*period_data)
                axes[1].scatter(indices, scores, c=colors_map.get(period, '#95A5A6'), label=period, s=50, alpha=0.7)
        
        axes[1].set_xlabel('歌曲索引', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('情感值', fontsize=12, fontweight='bold')
        axes[1].set_title('随机50首歌曲情感值散点图', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[1].legend()
        axes[1].grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_detailed_sentiment_comparison(self, sentiment_results, output_path='utils/visualization/detailed_sentiment_comparison.png'):
        """绘制详细的情感对比图"""
        if not sentiment_results:
            return None
        
        # 按时期分组
        period_data = {}
        for result in sentiment_results:
            period = result.get('period', 'Unknown')
            if period not in period_data:
                period_data[period] = []
            period_data[period].append(result['sentiment_score'])
        
        if not period_data:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # 使用箱线图显示各时期的分布
        periods = list(period_data.keys())
        data_to_plot = [period_data[p] for p in periods if p in period_data and period_data[p]]
        
        if not data_to_plot:
            return None
        
        box_plot = ax.boxplot(data_to_plot, labels=periods, patch_artist=True)
        
        # 设置箱线图颜色
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('时期', fontsize=12, fontweight='bold')
        ax.set_ylabel('情感值', fontsize=12, fontweight='bold')
        ax.set_title('各时期情感值分布箱线图', fontsize=16, fontweight='bold', color='#2C3E50')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_random_50_songs_sentiment_divided(self, divided_results, output_path='utils/visualization/random_50_songs_sentiment_divided.png'):
        """绘制分成5个表格的随机50首歌曲情感分析图"""
        if not divided_results or len(divided_results) == 0:
            return None
        
        # 创建5个子图
        fig, axes = plt.subplots(5, 1, figsize=(16, 20), facecolor='white')
        fig.suptitle('随机50首薛之谦歌曲情感分析（分组展示）', fontsize=18, fontweight='bold', color='#2C3E50')
        
        # 颜色方案
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        
        for idx, group in enumerate(divided_results):
            if idx >= 5:  # 最多显示5组
                break
            
            ax = axes[idx]
            
            if not group:
                ax.text(0.5, 0.5, f'第{idx+1}组：无数据', horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'第{idx+1}组 (无数据)', fontsize=14, fontweight='bold')
                continue
            
            # 提取数据
            song_names = [item['song_name'] for item in group]
            sentiment_scores = [item['sentiment_score'] for item in group]
            positive_scores = [item['positive_score'] for item in group]
            negative_scores = [item['negative_score'] for item in group]
            
            # 为歌曲名称添加索引以避免重复
            indexed_names = [f"{name}" for i, name in enumerate(song_names)]
            
            # 设置x轴
            x_pos = range(len(indexed_names))
            
            # 绘制情感值线图
            ax2 = ax.twinx()  # 创建第二个y轴用于显示积极/消极分数
            
            # 绘制情感值
            line1 = ax.plot(x_pos, sentiment_scores, 'o-', color=colors[idx % len(colors)], 
                           linewidth=2, markersize=6, label='情感值')
            
            # 绘制积极和消极分数
            line2 = ax2.bar([x - 0.2 for x in x_pos], positive_scores, 0.2, 
                           alpha=0.6, color='#2ECC71', label='积极分数')
            line3 = ax2.bar([x for x in x_pos], negative_scores, 0.2, 
                           alpha=0.6, color='#E74C3C', label='消极分数')
            
            # 设置x轴标签，避免重叠
            ax.set_xticks(x_pos)
            ax.set_xticklabels(indexed_names, rotation=75, ha='right', fontsize=7)
            
            # 调整布局以避免标签重叠
            plt.setp(ax.get_xticklabels(), rotation=75, ha='right', fontsize=7)
            
            # 设置标签和标题
            ax.set_ylabel('情感值', fontsize=12, fontweight='bold')
            ax2.set_ylabel('积极/消极分数', fontsize=12, fontweight='bold')
            ax.set_title(f'第{idx+1}组 - 10首歌曲情感分析', fontsize=14, fontweight='bold')
            
            # 添加网格
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 添加图例
            lines = line1 + [line2] + [line3]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            # 设置y轴范围
            ax.set_ylim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_lda_topics(self, topics, output_path='utils/visualization/lda_topics.png'):
        """绘制LDA主题模型结果"""
        if not topics:
            return None
        
        n_topics = len(topics)
        if n_topics == 0:
            return None
        
        # 计算需要的子图数量（每行最多3个主题）
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), facecolor='white')
        if n_topics == 1:
            axes = [axes]
        elif n_rows == 1 and n_topics < 3:
            axes = axes if n_topics > 1 else [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        fig.suptitle('LDA主题模型分析结果', fontsize=16, fontweight='bold', color='#2C3E50')
        
        # 选择颜色方案
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        
        for i, topic in enumerate(topics):
            if i < len(axes):
                top_words = topic['top_words'][:10]  # 取前10个词
                weights = topic['weights'][:10]
                
                bars = axes[i].barh(range(len(top_words)), weights, color=colors[i % len(colors)], alpha=0.8)
                axes[i].set_yticks(range(len(top_words)))
                axes[i].set_yticklabels(top_words, fontsize=9)
                axes[i].set_xlabel('权重', fontsize=10)
                axes[i].set_title(f'主题 {topic["topic_id"] + 1}', fontsize=12, fontweight='bold')
                axes[i].invert_yaxis()  # 使最高权重的词在顶部
                
                # 添加数值标签
                for bar, weight in zip(bars, weights):
                    axes[i].text(weight + max(weights)*0.01, bar.get_y() + bar.get_height()/2, 
                                f'{weight:.3f}', va='center', fontsize=8)
        
        # 隐藏多余的子图
        for i in range(n_topics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig
    
    def plot_co_occurrence_network(self, co_occurrence_data, output_path='utils/visualization/co_occurrence_network.png'):
        """绘制共现网络图"""
        if not co_occurrence_data:
            return None
        
        try:
            import networkx as nx
            
            # 从数据中提取节点和边
            nodes = co_occurrence_data.get('nodes', [])
            edges = co_occurrence_data.get('edges', [])
            
            if not nodes or not edges:
                print("共现网络数据不足，无法生成可视化")
                return None
            
            # 限制节点和边的数量以避免图过于复杂
            nodes = nodes[:30]  # 限制节点数量
            edges = [(u, v) for u, v in edges if u in nodes and v in nodes][:100]  # 限制边数量
            
            if not edges:
                print("筛选后的边数据不足，无法生成可视化")
                return None
            
            # 创建图
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            plt.figure(figsize=(14, 10), facecolor='black')  # 黑色背景
            
            # 使用kamada_kawai_layout，通常比spring_layout更快
            pos = nx.kamada_kawai_layout(G)
            
            # 为节点选择高级配色方案
            node_colors = [plt.cm.viridis(G.degree(node) / max(dict(G.degree()).values())) for node in G.nodes()]
            
            # 绘制节点
            node_sizes = [max(100, G.degree(node) * 80) for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                                 alpha=0.8, edgecolors='white', linewidths=1.5)
            
            # 绘制边，使用对比色
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6, edge_color='lightgray', 
                                 edge_cmap=plt.cm.Blues, arrows=False)
            
            # 绘制标签（只显示度数较高的节点标签）
            high_degree_nodes = {node: node for node in G.nodes() 
                                if G.degree(node) >= 3}  # 只显示度数>=3的节点标签，提高对比度
            nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, font_size=10, 
                                 font_family='SimHei', font_color='white', font_weight='bold')
            
            plt.title('歌词词汇共现网络', fontsize=16, fontweight='bold', color='white')
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')  # 黑色背景输出
            
            plt.close()  # 移除plt.show()，避免在批处理时阻塞
            
        except Exception as e:
            print(f"生成共现网络可视化时出错: {e}")
            return None
        
        return G
    
    def plot_comparison_with_jaychou(self, comparison_result, output_path=None):
        """绘制与周杰伦的对比分析图"""
        if not comparison_result:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle('薛之谦 vs 周杰伦 歌词对比分析', fontsize=18, fontweight='bold', color='#2C3E50')
        
        # 1. 歌曲数量对比
        xzq_count = comparison_result['xzq_song_count']
        jz_count = comparison_result['jz_song_count']
        
        axes[0, 0].bar(['薛之谦', '周杰伦'], [xzq_count, jz_count], 
                      color=['#E74C3C', '#3498DB'], alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('歌曲数量对比', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0, 0].set_ylabel('歌曲数量', fontsize=12)
        
        # 添加数值标签
        for i, v in enumerate([xzq_count, jz_count]):
            axes[0, 0].text(i, v + max([xzq_count, jz_count])*0.01, str(v), 
                           ha='center', va='bottom', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. 情感值对比
        xzq_sentiment = comparison_result['xzq_avg_sentiment']
        jz_sentiment = comparison_result['jz_avg_sentiment']
        xzq_std = comparison_result['xzq_sentiment_std']
        jz_std = comparison_result['jz_sentiment_std']
        
        x_pos = [0, 1]
        values = [xzq_sentiment, jz_sentiment]
        errors = [xzq_std, jz_std]
        
        bars = axes[0, 1].bar(['薛之谦', '周杰伦'], values, yerr=errors, 
                             color=['#E74C3C', '#3498DB'], alpha=0.8, edgecolor='black', linewidth=0.5, capsize=5)
        axes[0, 1].set_title('平均情感值对比', fontsize=14, fontweight='bold', color='#2C3E50')
        axes[0, 1].set_ylabel('平均情感值', fontsize=12)
        axes[0, 1].set_ylim(0, 1)
        
        # 添加数值标签
        for i, (v, e) in enumerate(zip(values, errors)):
            axes[0, 1].text(i, v + e + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. 薛之谦独特高频词
        xzq_unique = list(comparison_result['xzq_unique_words'].items())[:10]
        if xzq_unique:
            words, freqs = zip(*xzq_unique)
            bars = axes[1, 0].barh(range(len(words)), freqs, color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[1, 0].set_yticks(range(len(words)))
            axes[1, 0].set_yticklabels(words, fontsize=10)
            axes[1, 0].set_xlabel('词频', fontsize=12)
            axes[1, 0].set_title('薛之谦独特高频词', fontsize=14, fontweight='bold', color='#2C3E50')
            axes[1, 0].invert_yaxis()
            
            # 添加数值标签
            for bar, freq in zip(bars, freqs):
                axes[1, 0].text(freq + max(freqs)*0.01, bar.get_y() + bar.get_height()/2, 
                               str(freq), va='center', fontsize=9, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('薛之谦独特高频词', fontsize=14, fontweight='bold', color='#2C3E50')
        
        # 4. 周杰伦独特高频词
        jz_unique = list(comparison_result['jz_unique_words'].items())[:10]
        if jz_unique:
            words, freqs = zip(*jz_unique)
            bars = axes[1, 1].barh(range(len(words)), freqs, color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[1, 1].set_yticks(range(len(words)))
            axes[1, 1].set_yticklabels(words, fontsize=10)
            axes[1, 1].set_xlabel('词频', fontsize=12)
            axes[1, 1].set_title('周杰伦独特高频词', fontsize=14, fontweight='bold', color='#2C3E50')
            axes[1, 1].invert_yaxis()
            
            # 添加数值标签
            for bar, freq in zip(bars, freqs):
                axes[1, 1].text(freq + max(freqs)*0.01, bar.get_y() + bar.get_height()/2, 
                               str(freq), va='center', fontsize=9, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('周杰伦独特高频词', fontsize=14, fontweight='bold', color='#2C3E50')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        return fig

if __name__ == "__main__":
    import json
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # 加载分析结果
    analysis_path = os.path.join(project_dir, "utils", "data", "analysis_results.json")
    
    if not os.path.exists(analysis_path):
        print(f"分析结果文件不存在: {analysis_path}")
        exit(1)
    
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
    
    # 创建可视化对象
    visualizer = LyricVisualizer()
    
    # 生成词云
    output_dir = os.path.join(project_dir, "utils", "visualization")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 词云图
    wordcloud_path = os.path.join(output_dir, "wordcloud.png")
    if 'all_words' in analysis_results['word_frequency']:
        visualizer.generate_wordcloud(analysis_results['word_frequency']['all_words'], wordcloud_path)
    else:
        # 如果没有all_words键，使用word_counts
        visualizer.generate_wordcloud(analysis_results['word_frequency'], wordcloud_path)
    
    # 高频词汇条形图
    top_words_path = os.path.join(output_dir, "top_words.png")
    if 'all_words' in analysis_results['word_frequency']:
        visualizer.plot_top_words(analysis_results['word_frequency']['all_words'], top_words_path)
    else:
        visualizer.plot_top_words(analysis_results['word_frequency'], top_words_path)
    
    # 情感趋势图
    sentiment_path = os.path.join(output_dir, "sentiment_analysis.png")
    if 'sentiment' in analysis_results:
        visualizer.plot_sentiment_trend(analysis_results['sentiment'], sentiment_path)
    
    # 主题分布图
    if 'topics' in analysis_results and analysis_results['topics']:
        topic_path = os.path.join(output_dir, "topic_distribution.png")
        visualizer.plot_topic_distribution(analysis_results['topics'], topic_path)
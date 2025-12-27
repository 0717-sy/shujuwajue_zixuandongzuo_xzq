import os
import sys
import json
import traceback
import numpy as np
from utils.preprocess import LyricsPreprocessor
from utils.analysis import LyricsAnalyzer
from utils.visualization import LyricVisualizer
from utils.comparison_analysis import ComparisonAnalyzer

def run_preprocessing():
    print("\n预处理模块")

    processor = LyricsPreprocessor()
    input_file = '薛之谦歌词分析/utils/data/xuezhiqian.json'
        
    lyrics_data = processor.load_lyrics_data(input_file)
    print(f'加载了 {len(lyrics_data)} 首歌曲')
        
     # 预处理数据
    processed_data = processor.preprocess_dataset(lyrics_data)
        
     # 创建输出目录
    output_dir = '薛之谦歌词分析/utils/data'
    os.makedirs(output_dir, exist_ok=True)
        
     # 保存预处理后的数据
    output_file = os.path.join(output_dir, '预处理.json')
    processor.save_processed_data(processed_data, output_file)
    print(f'预处理数据已保存到 {output_file}')

    word_freq = processor.get_word_frequency(processed_data, top_k=20)
    print('\n前20个高频词:')
    for word, freq in word_freq[:20]:
        print(f'{word}: {freq}')
        
    return processed_data

def run_analysis(processed_data):
    print("\n分析模块")
            
    analyzer = LyricsAnalyzer()
    analyzer.data = processed_data  # 直接赋值预处理后的数据
        
    print(f'加载了 {len(analyzer.data)} 首歌曲的数据')
        
    print("\n1. 词频分析")
    word_freq = analyzer.word_frequency_analysis(top_k=200)
    print("前20个高频词:", [w for w, c in word_freq['all_words'][:10]])

        
    print("\n2. 歌词长度分析")
    length_df = analyzer.lyrics_length_analysis()
    print(f"平均词数: {length_df['word_count'].mean():.2f}")
        
    print("\n3. 词汇丰富度分析")
    ttr, types, tokens = analyzer.vocabulary_richness_analysis()
    print(f"型例比: {ttr:.4f}, 词汇类型数: {types}, 词汇标记数: {tokens}")
        
    print("\n4. 情感分析")
    sentiment_data = analyzer.sentiment_analysis()
    avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_data]) if sentiment_data else 0
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
        print(f"共现网络节点数: {len(G.nodes())}, 边数: {len(G.edges())}")
        
    print("\n7. 风格演进分析")
    df, yearly_stats = analyzer.style_evolution_analysis()
    print("按年份的情感、词汇丰富度、词数统计:")
    # 确保列名是中文
    if hasattr(yearly_stats, 'columns'):
        yearly_stats.columns = ['年份', '平均情感值', '平均词汇丰富度', '平均词数']
    print(yearly_stats.head() if hasattr(yearly_stats, 'head') else yearly_stats)
        
    print("\n8. 新词涌现分析")
    new_words = analyzer.new_word_emergence_analysis()
    for period, words in list(new_words.items())[:3]:  # 只显示前3个时期
        print(f"{period} 特有词: {[w[0] for w in words[:5]]}")
        
    print("\n9. 时期对比分析")
    period_stats, period_data = analyzer.period_comparison_analysis()
    for period, stats in period_stats.items():
        print(f"{period}: 歌曲数 {stats['song_count']}, 平均情感值 {stats['avg_sentiment']:.3f}")
        
    print("\n10. 比喻词分析")
    metaphor_patterns = analyzer.metaphor_analysis()
    print(f"发现 {len(metaphor_patterns)} 个比喻模式")
        
    print("\n11. 否定词分析")
    negation_analysis = analyzer.negation_analysis()
    print(f"否定词总数: {len(negation_analysis['overall_freq'])}")
        
    print("\n12. 句长与复杂度分析")
    sentence_complexity = analyzer.sentence_complexity_analysis()
        
    # 处理无法序列化的对象
    processed_co_occurrence = None
    if co_occurrence_result:
        G = co_occurrence_result.get('graph')
        if G:
            processed_co_occurrence = {
                'nodes': list(G.nodes()),
                'edges': list(G.edges()),
                'centrality': co_occurrence_result.get('centrality', {}),
                'betweenness': co_occurrence_result.get('betweenness', {}),
                'clustering': co_occurrence_result.get('clustering', {}),
            }
        else:
            processed_co_occurrence = co_occurrence_result
    
    # 保存分析结果
    analysis_results = {
        'word_frequency': word_freq,
        'lyrics_length': length_df.to_dict() if 'length_df' in locals() else {},
        'vocabulary_richness': {'ttr': ttr, 'types': types, 'tokens': tokens},
        'sentiment': sentiment_data,
        'topics': topics if 'topics' in locals() and topics else [],
        'co_occurrence': processed_co_occurrence,
        'yearly_stats': yearly_stats.to_dict() if 'yearly_stats' in locals() and hasattr(yearly_stats, 'to_dict') else {},
        'new_words': new_words if 'new_words' in locals() else {},
        'period_stats': period_stats if 'period_stats' in locals() else {},
        'metaphor_patterns': metaphor_patterns if 'metaphor_patterns' in locals() else [],
        'negation_analysis': negation_analysis if 'negation_analysis' in locals() else {},
        'sentence_complexity': sentence_complexity if 'sentence_complexity' in locals() else []
    }
        
    # 创建结果目录
    results_dir = '薛之谦歌词分析/utils/data'
    os.makedirs(results_dir, exist_ok=True)
        
    # 保存分析结果
    results_file = os.path.join(results_dir, 'analysis_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存到 {results_file}")
        
    return analysis_results

def run_visualization():
    print("\n可视化模块")

     # 加载分析结果
    results_file = '薛之谦歌词分析/utils/data/analysis_results.json'
        
    with open(results_file, 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
        
    # 创建可视化对象
    visualizer = LyricVisualizer()
        
    # 创建可视化输出目录
    output_dir = '薛之谦歌词分析/utils/visualization'
    os.makedirs(output_dir, exist_ok=True)
        
    # 生成词云图
    if 'word_frequency' in analysis_results and 'all_words' in analysis_results['word_frequency']:
        word_counts = {word: count for word, count in analysis_results['word_frequency']['all_words'][:200]}
        wordcloud_path = os.path.join(output_dir, "词云图.png")
        visualizer.generate_wordcloud(word_counts, wordcloud_path)
        print(f"词云图已生成")
        
    # 高频词条形图
    if 'word_frequency' in analysis_results and 'all_words' in analysis_results['word_frequency']:
        top_words = analysis_results['word_frequency']['all_words'][:20]
        top_words_path = os.path.join(output_dir, "高频词条形图.png")
        visualizer.plot_top_words(top_words, top_words_path)
        print(f"高频词条形图已生成")

    # 情感分析图
    if 'sentiment' in analysis_results:
        sentiment_path = os.path.join(output_dir, "情感分析图.png")
        visualizer.plot_sentiment_trend(analysis_results['sentiment'], sentiment_path)
        print(f"情感分析图已生成")
        
    # 主题分布图
    if 'topics' in analysis_results and analysis_results['topics']:
        topic_path = os.path.join(output_dir, "主题分布图.png")
        visualizer.plot_topic_distribution(analysis_results['topics'], topic_path)
        print(f"主题分布图已生成")
        
    # 按词性分类的词云图
    if 'word_frequency' in analysis_results:
        pos_words_dict = {
            'nouns': analysis_results['word_frequency'].get('nouns', []),
            'verbs': analysis_results['word_frequency'].get('verbs', []),
            'adjectives': analysis_results['word_frequency'].get('adjectives', []),
            'negations': analysis_results['word_frequency'].get('negations', [])
        }
        pos_wordcloud_path = os.path.join(output_dir, "词性词云图.png")
        visualizer.generate_pos_wordcloud(pos_words_dict, pos_wordcloud_path)
        print(f"词性词云图已生成")
        
    # 详细词频分析图
    if 'word_frequency' in analysis_results:
        visualizer.plot_detailed_word_frequency(analysis_results['word_frequency'], 
                                        os.path.join(output_dir, "详细词频分析图.png"))
        print("详细词频分析图已生成")
        
    # 比喻词分析图
    if 'metaphor_patterns' in analysis_results:
        visualizer.plot_metaphor_analysis(analysis_results['metaphor_patterns'], 
                                os.path.join(output_dir, "比喻词分析图.png"))
        print("比喻词分析图已生成")
        
    # 句子复杂度分析图
    if 'sentence_complexity' in analysis_results:
        visualizer.plot_sentence_complexity(analysis_results['sentence_complexity'], 
                                os.path.join(output_dir, "句子复杂度分析图.png"))
        print("句子复杂度分析图已生成")

    # 时期对比分析图
    if 'period_stats' in analysis_results:
        visualizer.plot_period_comparison(analysis_results['period_stats'], 
                                os.path.join(output_dir, "时期对比分析图.png"))
        print("时期对比分析图已生成")
        
    # 时间段情感分析图
    if 'sentiment' in analysis_results:
        visualizer.plot_time_period_sentiment_analysis(analysis_results['sentiment'], 
                                            os.path.join(output_dir, "时间段情感分析图.png"))
        print("时间段情感分析图已生成")

    # 共现网络分析图
    if 'co_occurrence' in analysis_results and analysis_results['co_occurrence']:
        co_occurrence_path = os.path.join(output_dir, "共现网络分析图.png")
        visualizer.plot_co_occurrence_network(analysis_results['co_occurrence'], co_occurrence_path)
        print(f"共现网络分析图已生成")

    return True

def run_comparison_analysis():
    print("\n对比分析模块")

    analyzer = ComparisonAnalyzer()
        
    # 检查数据文件
    xzq_file = '薛之谦歌词分析/utils/data/xuezhiqian_lyrics_valid_final.json'
    jaychou_file = '薛之谦歌词分析/utils/jaychou_lyrics.txt'
        
    # 加载薛之谦数据
    with open(xzq_file, 'r', encoding='utf-8') as f:
        xzq_data = json.load(f)
        
    # 预处理薛之谦数据
    xzq_processed = analyzer.preprocess_lyrics(xzq_data)
    print(f'薛之谦数据预处理完成，共{len(xzq_processed)}首歌曲')
        
    # 加载周杰伦数据
    jaychou_raw = analyzer.load_jaychou_lyrics(jaychou_file)
    jaychou_processed = analyzer.preprocess_lyrics(jaychou_raw)
    print(f'周杰伦数据预处理完成，共{len(jaychou_processed)}首歌曲')
        
    # 独特词分析
    print('独特词分析')
    unique_analysis = analyzer.unique_word_analysis(xzq_processed, jaychou_processed)
        
    # 保存对比分析结果
    comparison_result = {
        'unique_analysis': unique_analysis,
        'xzq_song_count': len(xzq_processed),
        'jz_song_count': len(jaychou_processed),
        'xzq_avg_sentiment': np.mean([analyzer.get_sentiment_score(song.get('cleaned_lyric', '')) for song in xzq_processed]) if xzq_processed else 0.5,
        'jz_avg_sentiment': np.mean([analyzer.get_sentiment_score(song.get('cleaned_lyric', '')) for song in jaychou_processed]) if jaychou_processed else 0.5,
        'xzq_sentiment_std': 0.1,  # 模拟值
        'jz_sentiment_std': 0.1,   # 模拟值
        'xzq_unique_words': dict(unique_analysis['xzq_unique'][:20]),
        'jz_unique_words': dict(unique_analysis['jaychou_unique'][:20])
    }
        
    # 保存对比分析结果
    output_dir = '薛之谦歌词分析/utils/data'
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, 'comparison_result.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)
    print(f'对比分析结果已保存到: {comparison_file}')
        
    # 输出结果
    print('\n薛之谦独特高频词 Top 10')
    for i, (word, freq) in enumerate(unique_analysis['xzq_unique'][:10], 1):
        print(f'{i:2d}. {word}: {freq}次')
        
    print('\n周杰伦独特高频词 Top 10 ')
    for i, (word, freq) in enumerate(unique_analysis['jaychou_unique'][:10], 1):
        print(f'{i:2d}. {word}: {freq}次')
        
    print('\n歌词特征与流行度相关性矩阵 ')

    visualizer = LyricVisualizer()
    comparison_viz_path = os.path.join('薛之谦歌词分析/utils/visualization', 'comparison_analysis.png')
    visualizer.plot_comparison_with_jaychou(comparison_result, comparison_viz_path)
    print(f'对比分析图已生成: {comparison_viz_path}')

def main():
    print("\n薛之谦歌词分析")

    os.makedirs('薛之谦歌词分析/utils/data', exist_ok=True)
    os.makedirs('薛之谦歌词分析/utils/visualization', exist_ok=True)
    
    # 运行预处理
    processed_data = run_preprocessing()
    if processed_data is None:
        print("预处理失败，程序终止")
        return
    
    # 运行分析
    analysis_results = run_analysis(processed_data)
    if analysis_results is None:
        print("分析失败，程序终止")
        return
    
    # 运行可视化
    success = run_visualization()
    if not success:
        print("可视化失败，程序终止")
        return
    
    # 运行对比分析
    run_comparison_analysis()

    print("结果文件已保存到 utils/data目录")
    print("可视化图表已保存到 utils/visualization目录")

if __name__ == "__main__":
    main()
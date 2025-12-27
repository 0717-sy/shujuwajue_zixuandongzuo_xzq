import jieba
import jieba.analyse
import re
import json
import os
from collections import Counter
import jieba.posseg as pseg

class LyricsPreprocessor:
    def __init__(self):
        # 定义薛之谦常用独特词汇，添加到jieba词典
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
            '钗头凤', '王子归来', '其实', '演员', '绅士', '配合', '分开时',
            '天外来物', '认真的雪', '意外', '刚好', '刚刚好', '动物世界'
        ]
        
        # 添加自定义词典
        for word in self.xzq_words:
            jieba.add_word(word)
        
        # 定义需要过滤的人名和合作者名
        self.names_to_filter = {
            '薛之谦', 'Xue Zhiqian', 'XueZhiqian', '谦谦', '谦哥',
            '罗志祥', '张杰', '胡彦斌', '迪丽热巴', '岳云鹏', '郭德纲', '于谦',
            '黄龄', '郁可唯', '刘惜君', '陈学冬', '张信哲', '品冠', '光良', '王力宏',
            '陶喆', '林俊杰', '王心凌', '张韶涵', '蔡依林', 'S.H.E', '罗志祥', '小S',
            '庾澄庆', '伊能静', '黄维德', '刘诗诗', '胡歌', '霍建华', '林心如', '陈乔恩',
            '明道', '陈小春', '应采儿', '郑中基', '杨千嬅', '许志安', '黄宗泽', '胡杏儿',
            '钟嘉欣', '马德钟', '郑嘉颖', '胡定燕', '周柏豪', '胡鸿钧', '胡诺言', '胡说八道',
            '周星驰', '吴孟达', '黄圣依', '杨子', '黄圣依', '杨子', '黄圣依', '杨子',
        }
        
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

    def clean_lyrics(self, lyrics):
        if not lyrics or lyrics.strip() == '':
            return ''
        
        # 移除时间戳
        lyrics = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', lyrics)
        lyrics = re.sub(r'\[\d{2}:\d{2}\]', '', lyrics)
        
        # 移除标记行（如[Intro], [Verse]等）
        lines = lyrics.split('\n')
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
            if any(keyword in line for keyword in ['作词', '作曲', '编曲', '制作', '演唱', '总监', '音频编辑', '监制', '统筹', '企划', '发行', '录音', '混音', '母带', '后期', '编配', '和声', '吉他', '贝斯', '鼓', '键盘', '弦乐', '管乐', '制作人']):
                continue
            # 跳过可能的人名标注（如：薛：，张：，丽日（薛：等）
            if re.search(r'[\u4e00-\u9fa5]+[：:]|\([^)]*[\u4e00-\u9fa5]+[：:]', line):
                continue
            # 跳过单独的标点符号行
            if re.match(r'^[\s\W]+$', line):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def segment_text(self, text, remove_stopwords=True, pos_filter=None, remove_names=True):
        if not text:
            return []
        
        # 使用jieba进行分词
        if pos_filter:
            # 使用词性标注进行分词
            words = pseg.cut(text)
            result = []
            for word, flag in words:
                word = word.strip()
                # 过滤标点符号和非中文字符
                if not word or self._is_punctuation_or_symbol(word):
                    continue
                if remove_names and word in self.names_to_filter:
                    continue
                if remove_stopwords and word in self.stopwords:
                    continue
                if pos_filter and flag[0] not in pos_filter:
                    continue
                result.append(word)
            return result
        else:
            words = jieba.lcut(text)
            # 过滤标点符号和非中文字符
            words = [word for word in words if word.strip() != '' and not self._is_punctuation_or_symbol(word)]
            if remove_names:
                words = [word for word in words if word not in self.names_to_filter]
            if remove_stopwords:
                words = [word for word in words if word not in self.stopwords]
            return words
    
    def _is_punctuation_or_symbol(self, word):
        if not word:
            return True
        # 检查是否全为标点符号或特殊符号
        for char in word:
            # 检查Unicode分类，标点符号和符号
            if char.isalnum() or '\u4e00' <= char <= '\u9fff':  # 字母数字或中文字符
                return False
        # 如果所有字符都是标点符号或特殊符号，则返回True
        import string
        import unicodedata
        for char in word:
            category = unicodedata.category(char)
            if not (category.startswith('P') or category.startswith('S') or char in string.punctuation):
                return False
        return True

    def preprocess_dataset(self, lyrics_data):
        # 首先去重，删除重复歌曲
        lyrics_data = self.remove_duplicates(lyrics_data)
        
        processed_data = []
        
        # 定义要删除的歌曲名（包含各种可能的写法）
        songs_to_remove = {'okok', '重返十七岁', '方的言', 'OKOK', '重返十七歲', '方的言'}
        
        for song in lyrics_data:
            song_name = song.get('song_name', '').strip()
            
            # 检查是否为需要删除的歌曲
            if song_name.lower() in [s.lower() for s in songs_to_remove]:
                continue  # 跳过这些歌曲
            
            processed_song = song.copy()
            
            # 将歌曲名小写统一为大写
            if song_name != song_name.upper():
                processed_song['song_name'] = song_name.upper()
            
            # 清洗歌词
            cleaned_lyrics = self.clean_lyrics(song.get('lyric', ''))
            processed_song['cleaned_lyric'] = cleaned_lyrics
            
            # 分词（包含人名过滤）
            words = self.segment_text(cleaned_lyrics, remove_names=True)
            processed_song['words'] = words
            
            # 按词性分词（包含人名过滤）
            noun_words = self.segment_text(cleaned_lyrics, pos_filter=['n'], remove_names=True)
            verb_words = self.segment_text(cleaned_lyrics, pos_filter=['v'], remove_names=True)
            adj_words = self.segment_text(cleaned_lyrics, pos_filter=['a', 'ad', 'an'], remove_names=True)
            
            processed_song['nouns'] = noun_words
            processed_song['verbs'] = verb_words
            processed_song['adjectives'] = adj_words
            
            processed_data.append(processed_song)
        
        return processed_data

    def preprocess_lyrics_list(self, lyrics_data):
        # 首先去重
        unique_data = self.remove_duplicates(lyrics_data)
        # 然后预处理
        return self.preprocess_dataset(unique_data)

    def get_word_frequency(self, lyrics_data, top_k=100):
        all_words = []
        for song in lyrics_data:
            words = song.get('words', [])
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        return word_freq.most_common(top_k)
    
    def remove_duplicates(self, lyrics_data):
        seen = set()
        unique_data = []
        
        for song in lyrics_data:
            song_name = song.get('song_name', '').strip().lower()
            lyric_content = song.get('lyric', '').strip()
            
            # 创建一个唯一的标识符，基于歌曲名和歌词内容
            identifier = f"{song_name}_{hash(lyric_content)}"
            
            if identifier not in seen:
                seen.add(identifier)
                unique_data.append(song)
            
        return unique_data

    def load_lyrics_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def save_processed_data(self, processed_data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    processor = LyricsPreprocessor()

    import os
    import sys

    possible_paths = [
        '薛之谦歌词分析/utils/data/xuezhiqian.json',  # 从项目根目录运行
        '薛之谦歌词分析/utils/data/xuezhiqian_lyrics_valid_final.json',  # 从项目根目录运行其他文件
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '薛之谦歌词分析','utils', 'data', 'xuezhiqian_lyrics_gegeci.json'),  # 完整路径
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '薛之谦歌词分析','utils', 'data', 'xuezhiqian_lyrics_valid_final.json')  # 完整路径
    ]
    
    lyrics_file = None
    for path in possible_paths:
        if os.path.exists(path):
            lyrics_file = path
            break
    
    if lyrics_file is None:
        print('未找到歌词数据文件，请确保数据文件存在于 utils/data/ 目录中')
        sys.exit(1)  # 退出程序
    
    if os.path.exists(lyrics_file):
        lyrics_data = processor.load_lyrics_data(lyrics_file)
        print(f'加载了 {len(lyrics_data)} 首歌曲')
        
        # 预处理数据
        processed_data = processor.preprocess_dataset(lyrics_data)
        
        # 保存预处理后的数据
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'data', 'preprocessed_lyrics.json')
        processor.save_processed_data(processed_data, output_path)
        print(f'预处理数据已保存到 {output_path}')
        
        # 输出词频统计示例
        word_freq = processor.get_word_frequency(processed_data, top_k=20)
        print('\n前20个高频词:')
        for word, freq in word_freq[:20]:
            print(f'{word}: {freq}')
    else:
        print(f'歌词文件不存在: {lyrics_file}')
        print('当前工作目录:', os.getcwd())
        print('可用的数据文件:')
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'data')
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.json'):
                    print(f'  - {file}')
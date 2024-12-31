import streamlit as st
import requests
from bs4 import BeautifulSoup
from collections import Counter
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import WordCloud as PyEchartsWordCloud, Bar, Pie, Line, Scatter, Funnel, EffectScatter
from streamlit_echarts import st_pyecharts

# 页面标题
st.title("文本分析与词云生成")


# 输入框，用户输入文章URL
url = st.text_input("请输入文章的URL:")


if url:
    # 请求URL抓取文本内容
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        soup = BeautifulSoup(response.text, 'html.parser')
        # 后续代码保持不变，继续提取文本

        # 提取文本内容，综合考虑多种可能包含文本的标签
        tags_to_extract = ['p', 'div', 'article', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
        paragraphs = []
        for tag in tags_to_extract:
            paragraphs.extend([element.get_text() for element in soup.find_all(tag)])
        text = ' '.join(paragraphs)

        # 对文本分词
        words = jieba.cut(text)
        word_list = [word for word in words if len(word) > 1]  # 过滤掉单个字符
        word_counts = Counter(word_list)


        # 统计词频并展示前20个词汇
        most_common_words = word_counts.most_common(20)
        st.write("词频排名前 20 的词汇:")
        for word, count in most_common_words:
            st.write(f"{word}: {count}")


        # 设置字体
        font_path = 'SimHei.ttf'  # 请确保路径正确
        my_font = fm.FontProperties(fname=font_path)


        # 图形筛选
        chart_type = st.sidebar.selectbox("选择图形类型:", ["词云", "柱状图", "饼图", "折线图", "散点图", "热力图", "面积图"])
        visualization_library = st.sidebar.selectbox("选择可视化库:", ["matplotlib", "seaborn", "Pyecharts"])


        # 根据选择的图形类型和可视化库绘制
        if chart_type == "词云":
            if visualization_library == "matplotlib":
                wordcloud = WordCloud(font_path=font_path, width=800, height=400).generate_from_frequencies(word_counts)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                # 使用 matplotlib 生成词云，seaborn 不直接支持词云，可进一步使用 seaborn 的风格设置
                wordcloud = WordCloud(font_path=font_path, width=800, height=400).generate_from_frequencies(word_counts)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                wordcloud = (
                    PyEchartsWordCloud()
                 .add("", [(word, count) for word, count in most_common_words], word_size_range=[20, 100])
                 .set_global_opts(title_opts=opts.TitleOpts(title="词云图"))
                )
                st_pyecharts(wordcloud)
        elif chart_type == "柱状图":
            if visualization_library == "matplotlib":
                plt.bar(*zip(*most_common_words))
                plt.xticks(rotation=45, fontproperties=my_font)  # 设置x轴标签字体
                plt.ylabel('频率', fontproperties=my_font)  # 设置y轴标签字体
                plt.title('高频词柱状图', fontproperties=my_font)  # 设置标题字体
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                import pandas as pd
                df = pd.DataFrame(most_common_words, columns=['words', 'counts'])
                ax = sns.barplot(x='words', y='counts', data=df)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontproperties=my_font)
                ax.set_ylabel('频率', fontproperties=my_font)
                ax.set_title('高频词柱状图', fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                bar_chart = (
                    Bar()
                 .add_xaxis([word for word, _ in most_common_words])
                 .add_yaxis("高频词柱状图", [count for _, count in most_common_words])
                 .set_global_opts(
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                        title_opts=opts.TitleOpts(title="高频词柱状图")
                    )
                )
                st_pyecharts(bar_chart)
        elif chart_type == "饼图":
            if visualization_library == "matplotlib":
                counts = [count for word, count in most_common_words]
                labels = [word for word, count in most_common_words]
                plt.pie(counts, labels=labels, autopct='%1.1f%%')
                for label in plt.gca().texts:
                    label.set_fontproperties(my_font)
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                import pandas as pd
                df = pd.DataFrame(most_common_words, columns=['words', 'counts'])
                plt.pie(df['counts'], labels=df['words'], autopct='%1.1f%%')
                plt.title('高频词饼图', fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                pie_chart = (
                    Pie()
                 .add("", [(word, count) for word, count in most_common_words])
                 .set_global_opts(title_opts=opts.TitleOpts(title="高频词饼图"))
                 .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
                )
                st_pyecharts(pie_chart)
        elif chart_type == "折线图":
            if visualization_library == "matplotlib":
                words, counts = zip(*most_common_words)
                plt.plot(words, counts)
                plt.xticks(rotation=45, fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                import pandas as pd
                df = pd.DataFrame(most_common_words, columns=['words', 'counts'])
                sns.lineplot(x='words', y='counts', data=df)
                plt.xticks(rotation=45, fontproperties=my_font)
                plt.title('高频词折线图', fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                line_chart = (
                    Line()
                 .add_xaxis([word for word, _ in most_common_words])
                 .add_yaxis("高频词折线图", [count for _, count in most_common_words])
                 .set_global_opts(
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                        title_opts=opts.TitleOpts(title="高频词折线图")
                    )
                )
                st_pyecharts(line_chart)
        elif chart_type == "散点图":
            if visualization_library == "matplotlib":
                words, counts = zip(*most_common_words)
                plt.scatter(range(len(words)), counts)  # 修改这里，使用索引作为x轴
                plt.xticks(range(len(words)), [word for word, _ in most_common_words], rotation=45, fontproperties=my_font)  # 为x轴添加对应的词
                plt.ylabel('频率', fontproperties=my_font)
                plt.title('高频词散点图', fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                import pandas as pd
                df = pd.DataFrame(most_common_words, columns=['words', 'counts'])
                sns.scatterplot(x='words', y='counts', data=df)
                plt.xticks(rotation=45, fontproperties=my_font)
                plt.title('高频词散点图', fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                scatter_chart = (
                    Scatter()
                 .add_xaxis([word for word, _ in most_common_words])
                 .add_yaxis("高频词散点图", [count for _, count in most_common_words])
                 .set_global_opts(
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                        title_opts=opts.TitleOpts(title="高频词散点图")
                    )
                )
                st_pyecharts(scatter_chart)
        elif chart_type == "面积图":
            if visualization_library == "matplotlib":
                words, counts = zip(*most_common_words)
                plt.fill_between(words, counts)
                plt.xticks(rotation=45, fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                import pandas as pd
                df = pd.DataFrame(most_common_words, columns=['words', 'counts'])
                sns.lineplot(x='words', y='counts', data=df, estimator=None)
                plt.fill_between(df['words'], df['counts'], alpha=0.3)
                plt.xticks(rotation=45, fontproperties=my_font)
                plt.title('高频词面积图', fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                area_chart = (
                    Line()
                 .add_xaxis([word for word, _ in most_common_words])
                 .add_yaxis("高频词面积图", [count for _, count in most_common_words], areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
                 .set_global_opts(
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                        title_opts=opts.TitleOpts(title="高频词面积图")
                    )
                )
                st_pyecharts(area_chart)
        elif chart_type == "热力图":
            if visualization_library == "matplotlib":
                # 准备热力图数据
                heatmap_data = np.array([[count for _, count in most_common_words]])
                plt.figure(figsize=(10, 5))
                sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', xticklabels=[word for word, _ in most_common_words], yticklabels=['频率'])
                plt.title('词频热力图', fontproperties=my_font)
                plt.xticks(rotation=45, fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "seaborn":
                import pandas as pd
                df = pd.DataFrame(most_common_words, columns=['words', 'counts'])
                heatmap_data = np.array([counts for _, counts in most_common_words]).reshape(1, -1)
                ax = sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', xticklabels=[word for word, _ in most_common_words], yticklabels=['频率'])
                ax.set_title('词频热力图', fontproperties=my_font)
                plt.xticks(rotation=45, fontproperties=my_font)
                st.pyplot(plt)
            elif visualization_library == "Pyecharts":
                # Pyecharts 不直接支持热力图，可以使用漏斗图或其他方式模拟
                funnel_chart = (
                    Funnel()
                 .add(
                        "词频",
                        [(word, count) for word, count in most_common_words],
                        label_opts=opts.LabelOpts(position="inside")
                    )
                 .set_global_opts(title_opts=opts.TitleOpts(title="词频热力图"))
                )
                st_pyecharts(funnel_chart)


    except Exception as e:
        st.error(f"抓取内容时出错: {e}")
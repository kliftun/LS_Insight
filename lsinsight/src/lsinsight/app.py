"""
A program for performing NLP on biblical texts.
"""
import importlib.metadata
import sys
from PySide6 import QtWidgets
import json
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QWidget,
                               QVBoxLayout, QPushButton, QDockWidget, QFileDialog, QMessageBox)
from pathlib import Path
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy


def onLoadFinished(ok):
    if not ok:
        print("Error loading the page")
    else:
        print("Page loaded successfully")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LS Insight™ Ver. 1.0a by Jacob Dickerson')
        self.setWindowIcon(
            QIcon('C:/Users/jaked/PycharmProjects/LS_Insight/LS_Insight_Non-QT/lsinsight/src/lsinsight/LS_ICON.png'))
        self.setMinimumSize(800, 600)
        self.initDocks()
        self.initMenuBar()

    def initDocks(self):
        # Create and add dock widgets
        self.createDockWidget("Text Input", self.createTextInputWidget(), Qt.LeftDockWidgetArea)
        self.createDockWidget("NMF Analysis Results", self.createNMFAnalysisWidget(), Qt.RightDockWidgetArea)
        self.createDockWidget("Topic Graph", self.createTopicGraphWidget(), Qt.BottomDockWidgetArea)
        self.createDockWidget("Dependency Chart", self.createDependencyChartWidget(), Qt.BottomDockWidgetArea)

    def createDockWidget(self, title, widget, area):
        dockWidget = QDockWidget(title, self)
        dockWidget.setWidget(widget)
        dockWidget.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(area, dockWidget)

    def createTextInputWidget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        analyzeButton = QPushButton("Analyze")
        analyzeButton.clicked.connect(self.onNMFAnalyzeClicked)
        layout.addWidget(self.text_edit)
        layout.addWidget(analyzeButton)
        widget.setLayout(layout)
        return widget

    def createNMFAnalysisWidget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.nmf_results_display = QTextEdit()
        self.nmf_results_display.setReadOnly(True)
        layout.addWidget(self.nmf_results_display)
        widget.setLayout(layout)
        return widget

    def createTopicGraphWidget(self):
        self.plotly_graph_viewer = QWebEngineView()
        self.configureWebEngineView(self.plotly_graph_viewer)
        return self.plotly_graph_viewer

    def createDependencyChartWidget(self):
        self.dependency_chart_viewer = QWebEngineView()
        self.configureWebEngineView(self.dependency_chart_viewer)
        return self.dependency_chart_viewer

    def configureWebEngineView(self, view):
        view.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        view.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        view.loadFinished.connect(onLoadFinished)

    def saveInstance(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Instance", "", "LS Insight Files (*.lsi);;All Files (*)",
                                                  options=options)
        if fileName:
            if not fileName.endswith('.lsi'):
                fileName += '.lsi'
            appState = {
                "textData": self.text_edit.toPlainText(),
                "nmfAnalysisResults": self.nmf_results_display.toPlainText(),  # Save NMF Results
                "topicGraphHtml": "",  # Initialize to empty string, to be filled later
                "dependencyChartHtml": ""  # Initialize to empty string, to be filled later
            }
            # Read and store the HTML content for Topic Graph
            try:
                with open("plotly_graph.html", 'r', encoding='utf-8') as file:
                    appState["topicGraphHtml"] = file.read()
            except Exception as e:
                print(f"Error reading Topic Graph HTML: {e}")
                appState["topicGraphHtml"] = ""

            # Read and store the HTML content for Dependency Chart
            try:
                with open("dependency_chart.html", 'r', encoding='utf-8') as file:
                    appState["dependencyChartHtml"] = file.read()
            except Exception as e:
                print(f"Error reading Dependency Chart HTML: {e}")
                appState["dependencyChartHtml"] = ""

            with open(fileName, 'w', encoding='utf-8') as file:
                json.dump(appState, file, indent=4)

    def loadInstance(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Instance", "", "LS Insight Files (*.lsi);;All Files (*)",
                                                  options=options)
        if fileName:
            with open(fileName, 'r', encoding='utf-8') as file:
                appState = json.load(file)
                self.text_edit.setPlainText(appState.get("textData", ""))
                self.nmf_results_display.setPlainText(appState.get("nmfAnalysisResults", ""))  # Restore NMF Results

                # Load and display the HTML content for Topic Graph and Dependency Chart
                topicGraphHtml = appState.get("topicGraphHtml", "")
                dependencyChartHtml = appState.get("dependencyChartHtml", "")

                if topicGraphHtml:
                    self.plotly_graph_viewer.setHtml(topicGraphHtml)
                if dependencyChartHtml:
                    self.dependency_chart_viewer.setHtml(dependencyChartHtml)

    def initMenuBar(self):
        menuBar = self.menuBar()
        # File Menu
        fileMenu = menuBar.addMenu("&File")

        # Save and Load Instance actions
        saveInstanceAction = QAction("&Save Instance", self)
        saveInstanceAction.triggered.connect(self.saveInstance)

        loadInstanceAction = QAction("&Load Instance", self)
        loadInstanceAction.triggered.connect(self.loadInstance)

        exitAction = QAction("&Exit", self)
        exitAction.triggered.connect(QApplication.instance().quit)

        fileMenu.addAction(saveInstanceAction)
        fileMenu.addAction(loadInstanceAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        # Help Menu
        helpMenu = menuBar.addMenu("&Help")
        aboutAction = QAction("&About", self)
        aboutAction.triggered.connect(self.aboutDialog)
        helpMenu.addAction(aboutAction)

    def aboutDialog(self):
        QMessageBox.about(self, "About LS Insight™",
                          "Lexio Sapientia Insight™ Ver. 1.0a \n\n "
                          "Copyright (C) 2024 Jacob Dickerson\n\n"
                          "An application for NMF Analysis "
                          "of english biblical texts written in Python, using NLTK and spaCy. \n\n"
                          "For third-party license information see LICENSES.txt in the program folder. \n\n"
                          "Copyright (C) 2024 Jacob Dickerson")

    def display_dependency_chart(self, html_content):
        # Method to display the dependency chart HTML content
        self.dependency_chart_viewer.setHtml(html_content)

    def display_plotly_graph(self, html_content):
        # Method to display the Plotly graph HTML content
        self.plotly_graph_viewer.setHtml(html_content)

    def performNMFAnalysis(self):
        # Extract the text input from the user
        text_input = self.text_edit.toPlainText()

        # Perform the NMF analysis (assuming this function returns two values)
        components, top_words_per_topic = perform_topic_modeling(text_input)

        # Prepare and display the content in the NMF Analysis Tab
        display_content = ""
        for topic_idx, top_words in enumerate(top_words_per_topic, start=1):
            display_content += f"Topic {topic_idx}:\n"
            for word, weight in top_words:
                display_content += f"    {word}: {weight:.4f}\n"
            display_content += "\n"  # Add spacing between topics

        # Update the UI widget with the results
        # Assuming there's a QTextEdit or QLabel widget for displaying the results
        self.nmf_results_display.setText(display_content)
        plot_topics_3d_interactive(components, top_words_per_topic, "plotly_graph.html")

    def generateDependencyChart(self):
        text_input = self.text_edit.toPlainText()
        generate_dependency_chart(text_input)

    def onNMFAnalyzeClicked(self):
        self.performNMFAnalysis()
        self.generateDependencyChart()
        # Assuming 'plotly_graph.html' is in the current working directory
        plotly_html_path = os.path.abspath("plotly_graph.html")
        # Replace backslashes with forward slashes
        plotly_file_url = f"file:///{plotly_html_path.replace(os.sep, '/')}"
        dep_html_path = os.path.abspath("dependency_chart.html")
        dep_html_url = f"file:///{dep_html_path.replace(os.sep, '/')}"
        self.dependency_chart_viewer.setUrl(QUrl(dep_html_url))
        self.plotly_graph_viewer.setUrl(QUrl(plotly_file_url))


def generate_dependency_chart(text_input):
    doc = nlp(text_input)
    sentspans = list(doc.sents)
    options = {'color': '#ffffff', 'bg': '#2E86C1', 'compact': True}
    html = displacy.render(sentspans, style="dep", options=options, page=True)

    output_path = Path('dependency_chart.html')
    output_path.open("w", encoding="utf-8").write(html)


def plot_topics_3d_interactive(model_components, top_words_per_topic, output_filename):
    # Perform PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    reduced_components = pca.fit_transform(model_components)

    fig = go.Figure()

    # Updated color palette with sophisticated colors
    colors = [
        'rgba(10, 132, 255, opacity)',  # Vivid Blue
        'rgba(255, 55, 95, opacity)',  # Vibrant Red
        'rgba(52, 199, 89, opacity)',  # Bright Green
        'rgba(171, 0, 255, opacity)',  # Electric Purple
        'rgba(255, 94, 0, opacity)',  # Sunset Orange
        'rgba(153, 0, 153, opacity)',  # Deep Magenta
        'rgba(0, 206, 209, opacity)',  # Turquoise Blue
        'rgba(255, 127, 80, opacity)',  # Coral Pink
        'rgba(255, 247, 0, opacity)',  # Lemon Yellow
        'rgba(25, 25, 112, opacity)',  # Midnight Blue
        'rgba(255, 0, 56, opacity)',  # Cherry Red
        'rgba(181, 126, 220, opacity)',  # Lavender Purple
        'rgba(0, 128, 128, opacity)',  # Teal Green
        'rgba(255, 204, 153, opacity)',  # Peachy Pink
        'rgba(75, 0, 130, opacity)',  # Indigo Blue
        'rgba(255, 213, 79, opacity)',  # Mustard Yellow
        'rgba(224, 17, 95, opacity)',  # Ruby Red
        'rgba(135, 206, 235, opacity)',  # Sky Blue
        'rgba(152, 255, 152, opacity)',  # Mint Green
        'rgba(218, 112, 214, opacity)',  # Orchid Purple
        'rgba(0, 255, 255, opacity)',  # Cyan
        'rgba(255, 165, 0, opacity)',  # Orange
        'rgba(255, 99, 71, opacity)',  # Tomato
        'rgba(255, 20, 147, opacity)',  # DeepPink
        'rgba(255, 0, 255, opacity)',  # Magenta
        'rgba(255, 105, 180, opacity)',  # Pink
        'rgba(238, 130, 238, opacity)',  # Violet
        'rgba(255, 140, 0, opacity)',  # DarkOrange
        'rgba(255, 215, 0, opacity)',  # Gold
        'rgba(0, 128, 0, opacity)',  # Green
        'rgba(255, 192, 203, opacity)',  # Pink
        # Add more colors as needed, replacing 'opacity' with actual opacity values later
    ]

    for i, component in enumerate(reduced_components):
        top_word_info = top_words_per_topic[i]

        # Calculate opacity to enhance visual differentiation; adjust as needed
        opacity = 0.6 + 0.4 * (i / len(reduced_components))  # Gradually increasing opacity
        color = colors[i % len(colors)].replace('opacity', str(opacity))
        legend_group = f"topic_{i + 1}"

        # Main topic point without text
        fig.add_trace(go.Scatter3d(x=[component[0]], y=[component[1]], z=[component[2]],
                                   mode='markers',
                                   marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey')),
                                   name=f'Topic {i + 1}',
                                   legendgroup=legend_group,
                                   hoverinfo='text'))

        # Calculate z-axis label offset for distancing the label higher on the x-axis
        z_label_offset = .3  # Adjust this value based on your dataset scale and visual preference
        label_z_position = component[2] + z_label_offset

        # Adding text trace for the label, positioned higher on the x-axis
        fig.add_trace(go.Scatter3d(x=[component[0]], y=[component[1]], z=[label_z_position],
                                   mode='text',
                                   text=f"Topic {i + 1} ",
                                   hoverinfo='none',
                                   legendgroup=legend_group,
                                   showlegend=False,
                                   textfont=dict(size=13, color="white")))

        # Word clusters with refined marker aesthetics
        for word, weight in top_word_info:
            word_position = component + np.random.normal(loc=0.0, scale=0.05, size=3)
            fig.add_trace(go.Scatter3d(x=[word_position[0]], y=[word_position[1]], z=[word_position[2]],
                                       mode='markers',
                                       marker=dict(size=8, color=color, line=dict(width=1, color='DarkSlateGrey')),
                                       text=f"{word} ({weight:.2f})",
                                       hoverinfo='text',
                                       legendgroup=legend_group,
                                       showlegend=False))

    # Refined layout for a sleek appearance
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='PCA 3',
        xaxis=dict(backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="gray",
                   showbackground=True,
                   zerolinecolor="gray", ),
        yaxis=dict(backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="gray",
                   showbackground=True,
                   zerolinecolor="gray"),
        zaxis=dict(backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="gray",
                   showbackground=True,
                   zerolinecolor="gray"),
    ),
                      paper_bgcolor="rgb(10, 10, 10)",
                      plot_bgcolor='rgb(10, 10, 10)',
                      font=dict(color="white"),
                      legend=dict(x=0, y=0, traceorder='normal', font=dict(family='sans-serif', size=12, color='white'))
                      )

    # Write the plot to an HTML file instead of returning it
    fig.write_html(output_filename, full_html=True, include_plotlyjs='cdn')


custom_stopwords_eng = [
    "a", "an", "the", "and", "or", "but", "for", "with", "in", "on", "at", "to", "from", "of", "by", "as", "is", "are",
    "were", "was", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "must", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "/", ",", ".", ":", ";", "(", ")",
    "[", "]", "{", "}", "'", "\"", "‘", "’", "“", "”", "he", "she", "it", "they", "we", "you", "that", "which", "who",
    "whom", "whose", "of", "to", "in", "for", "on", "with", "by", "at", "from", "into", "onto", "upon", "among",
    "between", "within", "without", "under", "over", "through", "during", "before", "after", "since", "about",
    "around", "against", "beside", "beyond", "above", "below", "throughout", "toward", "across", "is", "am", "are",
    "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "can", "could", "must", "good", "bad", "well", "better", "best", "often", "sometimes", "always",
    "never", "many", "few", "much", "little", "more", "less", "most", "least", "thou", "thee", "thine", "thy", "ye",
    "hath", "hast", "art", "didst", "doth", "wast", "were", "whence", "whom", "wilt"
]

nlp = spacy.load("en_core_web_trf")
print(spacy.info('en_core_web_trf'))

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tagged_tokens = nltk.pos_tag(tokens)

    english_stopwords = set(stopwords.words('english'))
    custom_stopwords_set = set(custom_stopwords_eng)
    all_stopwords_set = custom_stopwords_set.union(english_stopwords)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in tagged_tokens if
              token.isalpha() and token not in all_stopwords_set]

    return tokens


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun (for consistency)


def perform_topic_modeling(text, num_topics=5):
    preprocessed_tokens = preprocess_text(text)
    # Join tokens back into a single string
    ntext = ' '.join(preprocessed_tokens)

    # Vectorize the text
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=.2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform([ntext])  # Ensure ntext is a string here

    # Apply NMF
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_model.fit(tfidf)

    # Extract the topics
    topics = []
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words_per_topic = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]  # Indices of top words for this topic
        top_features = [feature_names[i] for i in top_features_ind]  # Top words
        weights = [topic[i] for i in top_features_ind]  # Weights of the top words

        # Prepare top words and their weights for visualization
        top_words_info = [(word, weight) for word, weight in zip(top_features, weights)]
        top_words_per_topic.append(top_words_info)

        # For textual representation, if needed
        topic_str = " + ".join([f"{weight:.3f}*{word}" for word, weight in zip(top_features, weights)])
        topics.append((topic_idx, topic_str))

    return nmf_model.components_, top_words_per_topic


def main():
    # Linux desktop environments use an app's .desktop file to integrate the app
    # in to their application menus. The .desktop file of this app will include
    # the StartupWMClass key, set to app's formal name. This helps associate the
    # app's windows to its menu item.
    #
    # For association to work, any windows of the app must have WMCLASS property
    # set to match the value set in app's desktop file. For PySide6, this is set
    # with setApplicationName().

    # Find the name of the module that was used to start the app
    app_module = sys.modules["__main__"].__package__
    # Retrieve the app's metadata
    metadata = importlib.metadata.metadata(app_module)

    QtWidgets.QApplication.setApplicationName(metadata["Formal-Name"])

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

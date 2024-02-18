import json
import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QWidget,
                               QVBoxLayout, QPushButton, QDockWidget, QFileDialog, QMessageBox)

from NMF3D import plot_topics_3d_interactive
from functions import perform_topic_modeling, generate_dependency_chart


def onLoadFinished(ok):
    if not ok:
        print("Error loading the page")
    else:
        print("Page loaded successfully")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LS Insight™ Ver. 1.0a by Jacob Dickerson')
        self.setWindowIcon(QIcon('LS_ICON.png'))
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
        analyzeButton.setIcon(QIcon('Analyze_ICON.png'))  # Ensure the icon exists
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

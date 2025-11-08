import logging
from pathlib import Path
import multiprocessing
import re
import html
import queue
import threading

from PySide6.QtCore import QThread, Signal, QObject, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout,
    QMessageBox, QApplication, QComboBox, QLabel, QTextBrowser, QToolButton
)

from abc import ABC, abstractmethod
from chat_llm import create_lm_studio_thread, create_openai_thread
from constants import TOOLTIPS
from vector_db_query import process_chunks_only_query
from config import get_config
from process_manager import get_process_manager

logger = logging.getLogger(__name__)


class SubmitStrategy(ABC):
    def __init__(self, tab):
        self.tab = tab

    @abstractmethod
    def submit(self, question: str, db_name: str) -> None:
        ...


class LMStudioStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        t = self.tab.llm_chat_thread = create_lm_studio_thread(question, db_name)

        t.response_signal.connect(self.tab.update_response_stream)
        t.error_signal.connect(self.tab.show_error_message)
        t.finished_signal.connect(self.tab.on_submission_finished)
        t.citations_signal.connect(self.tab.display_citations_in_widget)
        t.start()


class ChatGPTStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        model_name = self.tab.model_source_combo.currentText()
        t = self.tab.llm_chat_thread = create_openai_thread(question, db_name, model_name)

        t.response_signal.connect(self.tab.update_response_stream)
        t.error_signal.connect(self.tab.show_error_message)
        t.finished_signal.connect(self.tab.on_submission_finished)
        t.citations_signal.connect(self.tab.display_citations_in_widget)
        t.start()


class ChunksOnlyStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        t = self.tab.database_query_thread = ChunksOnlyThread(question, db_name)
        t.chunks_ready.connect(self.tab.display_chunks)
        t.finished.connect(self.tab.on_database_query_finished)
        t.start()


class ChunksOnlyThread(QThread):
    chunks_ready = Signal(str)

    def __init__(self, query, database_name):
        super().__init__()
        self.query = query
        self.database_name = database_name
        self.process = None
        self.process_lock = threading.Lock()

    def run(self):
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()

        try:
            with self.process_lock:
                self.process = ctx.Process(
                    target=process_chunks_only_query,
                    args=(self.database_name, self.query, result_queue)
                )
                get_process_manager().register(self.process)
                self.process.start()

            try:
                result = result_queue.get(timeout=30)
                self.chunks_ready.emit(result)
            except queue.Empty:
                logger.error("Query timed out after 60 seconds")
                self.chunks_ready.emit("Error: Query timed out after 60 seconds. Please try a simpler query or check your database.")
            except Exception as e:
                logger.error(f"Error getting result from queue: {e}")
                self.chunks_ready.emit(f"Error: Failed to retrieve database response - {str(e)}")

            with self.process_lock:
                if self.process and self.process.is_alive():
                    self.process.join(timeout=2)
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=1)
                        if self.process.is_alive():
                            try:
                                self.process.kill()
                                self.process.join(timeout=1)
                            except Exception as e:
                                logger.error(f"Failed to kill process: {e}")
                    
                if self.process:
                    get_process_manager().unregister(self.process)
                    self.process = None

        except Exception as e:
            logger.exception(f"Error in chunks only thread: {e}")
            self.chunks_ready.emit(f"Error querying database: {str(e)}")
            with self.process_lock:
                if self.process:
                    try:
                        if self.process.is_alive():
                            self.process.terminate()
                            self.process.join(timeout=1)
                            if self.process.is_alive():
                                self.process.kill()
                                self.process.join(timeout=1)
                        get_process_manager().unregister(self.process)
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup: {cleanup_error}")
                    finally:
                        self.process = None

    def stop(self):
        with self.process_lock:
            if self.process:
                try:
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=2)
                        if self.process.is_alive():
                            self.process.kill()
                            self.process.join(timeout=1)
                    get_process_manager().unregister(self.process)
                except Exception as e:
                    logger.warning(f"Error stopping process: {e}")
                finally:
                    self.process = None


class GuiSignals(QObject):
    response_signal = Signal(str)
    citations_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()


class CustomTextBrowser(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(QDesktopServices.openUrl)


class RefreshingComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_tab = parent

    def showPopup(self):
        if self.parent_tab:
            current_selection = self.currentText()
            self.parent_tab.refresh_database_list()
            index = self.findText(current_selection)
            if index >= 0:
                self.setCurrentIndex(index)
        super().showPopup()


class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.config_path = Path(__file__).resolve().parent / 'config.yaml'
        self.llm_chat_thread = None
        self.gui_signals = GuiSignals()
        self.database_query_thread = None
        self.raw_response = ""
        self.citations_block = ""
        self.initWidgets()

    def initWidgets(self):
        layout = QVBoxLayout(self)

        self.response_widget = CustomTextBrowser()
        self.response_widget.setOpenExternalLinks(True)
        layout.addWidget(self.response_widget, 5)

        hbox1 = QHBoxLayout()

        self.database_pulldown = RefreshingComboBox(self)
        self.database_pulldown.addItems(self.load_created_databases())
        self.database_pulldown.setToolTip(TOOLTIPS["DATABASE_SELECT"])
        hbox1.addWidget(self.database_pulldown)

        self.model_source_combo = QComboBox()
        self.model_source_combo.setToolTip(TOOLTIPS["MODEL_BACKEND_SELECT"])
        self.model_source_combo.addItems([
            "LM Studio",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4o",
        ])

        self.model_source_combo.setCurrentText("LM Studio")
        hbox1.addWidget(self.model_source_combo)

        self.query_settings_btn = QToolButton()
        self.query_settings_btn.setText("⚙")
        self.query_settings_btn.setToolTip("Open Database Query settings")
        self.query_settings_btn.clicked.connect(self._open_settings_dialog)
        hbox1.addWidget(self.query_settings_btn)

        layout.addLayout(hbox1)

        self.text_input = QTextEdit()
        self.text_input.setToolTip(TOOLTIPS["QUESTION_INPUT"])
        layout.addWidget(self.text_input, 1)

        hbox2 = QHBoxLayout()

        self.copy_response_button = QPushButton("Copy Response")
        self.copy_response_button.setToolTip(TOOLTIPS["COPY_RESPONSE"])
        self.copy_response_button.clicked.connect(self.on_copy_response_clicked)
        hbox2.addWidget(self.copy_response_button)

        self.chunks_only_checkbox = QCheckBox("Chunks Only")
        self.chunks_only_checkbox.setToolTip(TOOLTIPS["CHUNKS_ONLY"])
        hbox2.addWidget(self.chunks_only_checkbox)

        self.submit_button = QPushButton("Submit Question")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        hbox2.addWidget(self.submit_button)

        layout.addLayout(hbox2)

    def _open_settings_dialog(self):
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "show_settings_dialog") and callable(w.show_settings_dialog):
                w.show_settings_dialog()
                break

    def load_created_databases(self):
        config = get_config()
        databases = list(config.created_databases.keys())
        return [db for db in databases if db != "user_manual"]

    def refresh_database_list(self):
        databases = self.load_created_databases()
        self.database_pulldown.clear()
        self.database_pulldown.addItems(databases)

    def _render_html(self):
        body = html.escape(self.raw_response).replace("\n", "<br>")
        body += self.citations_block
        self.response_widget.setHtml(body)
        self.response_widget.verticalScrollBar().setValue(
            self.response_widget.verticalScrollBar().maximum()
        )

    def on_submit_button_clicked(self):
        self.response_widget.clear()
        self.raw_response = ""
        self.citations_block = ""
        self.submit_button.setDisabled(True)

        user_question = self.text_input.toPlainText().strip()
        if not user_question:
            QMessageBox.warning(self, "Error", "Please enter a question.")
            self.submit_button.setDisabled(False)
            return

        selected_database = self.database_pulldown.currentText()
        if not selected_database:
            QMessageBox.warning(self, "Error", "Please select a vector database.")
            self.submit_button.setDisabled(False)
            return

        if self.chunks_only_checkbox.isChecked():
            strategy = ChunksOnlyStrategy(self)
        else:
            source = self.model_source_combo.currentText()
            if source == "LM Studio":
                strategy = LMStudioStrategy(self)
            else:
                strategy = ChatGPTStrategy(self)

        try:
            strategy.submit(user_question, selected_database)
        except Exception as e:
            logger.exception("Submission failed: %s", e)
            self.show_error_message(str(e))
            self.submit_button.setDisabled(False)

    def display_chunks(self, chunks):
        self.response_widget.setPlainText(chunks)

    def on_database_query_finished(self):
        self.submit_button.setDisabled(False)

    def display_citations_in_widget(self, citations):
        if citations:
            self.citations_block = f"<br><br>Citation Links:{citations}"
        else:
            self.citations_block = "<br><br>No citations found."
        self._render_html()

    def on_copy_response_clicked(self):
        clipboard = QApplication.clipboard()
        response_text = self.response_widget.toPlainText()
        if response_text:
            clipboard.setText(response_text)
            QMessageBox.information(self, "Information", "Response copied to clipboard.")
        else:
            QMessageBox.warning(self, "Warning", "No response to copy.")

    def update_response_stream(self, response_chunk):
        self.raw_response += response_chunk
        self._render_html()

    def show_error_message(self, error_message):
        QMessageBox.warning(self, "Error", error_message)
        self.submit_button.setDisabled(False)

    def on_submission_finished(self):
        self.submit_button.setDisabled(False)

    def cleanup(self):
        if self.database_query_thread and self.database_query_thread.isRunning():
            try:
                if hasattr(self.database_query_thread, 'stop'):
                    self.database_query_thread.stop()
                self.database_query_thread.wait(1000)
            except Exception as e:
                logger.warning(f"Thread cleanup error: {e}")
            self.database_query_thread.wait()
        if self.llm_chat_thread and self.llm_chat_thread.isRunning():
            self.llm_chat_thread.wait()
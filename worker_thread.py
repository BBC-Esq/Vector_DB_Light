# worker_thread.py
from PySide6.QtCore import QThread, Signal
import logging

logger = logging.getLogger(__name__)

class VectorDBWorker(QThread):
    finished = Signal(bool, str)  # success, message
    progress = Signal(str)        # status updates

    def __init__(self, database_name, parent=None):
        super().__init__(parent)
        self.database_name = database_name

    def run(self):
        try:
            # Import here to avoid issues
            from vector_db_creator import CreateVectorDB
            
            self.progress.emit("Starting database creation...")
            create_vector_db = CreateVectorDB(database_name=self.database_name)
            create_vector_db.run()
            self.finished.emit(True, "Database created successfully")
            
        except Exception as e:
            logger.exception("Database creation failed")
            self.finished.emit(False, str(e))
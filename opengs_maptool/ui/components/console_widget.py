import asyncio
from html import escape
import opengs_maptool.config as config
from opengs_maptool.context import ApplicationContext
from opengs_maptool.services.command_service import execute_command_string, serialize_command
from opengs_maptool.models.command_response import CommandResponse
from opengs_maptool.models.message import Message, MessageAuthor, MessageType
from opengs_maptool.ui.modals.error_modal import ErrorModal
from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import QFileDialog, QTextEdit, QVBoxLayout, QWidget, QLineEdit, QPushButton, QHBoxLayout, QSizePolicy

class ConsoleWidget(QWidget):
    def __init__(self, context: ApplicationContext, main_window):
        super().__init__()
        self._context = context
        self._main_window = main_window

        monospace_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

        layout = QVBoxLayout()

        # Output log (read-only)
        self._output = QTextEdit()
        self._output.setFont(monospace_font)
        self._output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("Type 'link.help' for a help link.")
        layout.addWidget(self._output)

        # Input line (for user commands)
        self._input = QLineEdit()
        self._input.setFont(monospace_font)
        self._input.setPlaceholderText("Enter command...")
        self._input.returnPressed.connect(self._submit_user_command)
        layout.addWidget(self._input)

        # Buttons
        self._btn_export_logs = QPushButton("Export Logs")
        self._btn_export_logs.clicked.connect(self._export_logs)

        self._btn_restore_logs = QPushButton("Restore Logs")
        self._btn_restore_logs.clicked.connect(self._restore_logs)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self._btn_export_logs)
        buttons_layout.addWidget(self._btn_restore_logs)
        layout.addLayout(buttons_layout)

        # Keep controls compact while allowing the output area to consume extra vertical space.
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)

        self.setLayout(layout)


    def print_message(self, message: Message):
        """
        Displays a message in the console widget.
        The text style varies depending on the author and the type of message.

        @param message: The message to display
        """
        message_text = message.text
        message_color = None

        match message.type:
            case MessageType.COMMAND:
                message_color = config.CONSOLE_COMMAND_COLOR
                if message.author == MessageAuthor.USER:
                    message_text = f"USER COMMAND >> {message_text}"
                elif message.author == MessageAuthor.SYSTEM: # Only used for testing purposes.
                    message_text = f"GUI SYSTEM COMMAND >> {message_text}"

            case MessageType.NORMAL:
                message_color = config.CONSOLE_NORMAL_COLOR
            case MessageType.INFO:
                message_color = config.CONSOLE_INFO_COLOR
            case MessageType.SUCCESS:
                message_color = config.CONSOLE_SUCCESS_COLOR
            case MessageType.WARNING:
                message_color = config.CONSOLE_WARNING_COLOR
            case MessageType.ERROR:
                message_color = config.CONSOLE_ERROR_COLOR


        if message_color is not None:
            # QTextEdit collapses '\n' in rich text; use <br> for visible line breaks.
            html_text = escape(message_text).replace("\n", "<br>")
            self._output.append(f"<span style='color: {message_color};'>{html_text}</span>")
        else:
            # Insert as plain text to avoid accidental HTML parsing.
            self._output.moveCursor(QTextCursor.MoveOperation.End)
            self._output.insertPlainText(message_text + "\n")

        self._output.verticalScrollBar().setValue(self._output.verticalScrollBar().maximum())


    def _submit_user_command(
            self, manual_command: str | None = None, author: MessageAuthor = MessageAuthor.USER
        ) -> asyncio.Task[CommandResponse]:
        """
        Sends and displays a command in the console.
        """
        if manual_command is None:
            user_message_text = self._input.text().strip()
            self._input.clear()
        else:
            user_message_text = manual_command.strip()

        if len(user_message_text) == 0:
            return

        command_message = self._context.console_controller.add_command_message(user_message_text, author)
        self.print_message(command_message) # Show message in widget

        loop = asyncio.get_event_loop()
        task = loop.create_task(execute_command_string(self._context, command_message.text))
        task.add_done_callback(self._process_command_response)
        return task

    def submit_system_command(self, command_segments: list[str]) -> asyncio.Task[CommandResponse]:
        """
        Only used for testing purposes.
        Sends and displays a command in the console.
        """
        command_text = serialize_command(command_segments)
        return self._submit_user_command(manual_command=command_text, author=MessageAuthor.SYSTEM)

    def _process_command_response(self, task: asyncio.Task[CommandResponse]) -> None:
        response = task.result()
        message = response.as_message()
        self._context.console_controller.add_response_message(
            message.text,
            message.type
        )
        self.print_message(message)

    def _export_logs(self):
        filename, _ = QFileDialog.getSaveFileName(
            None, "Export console history", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return

        self._context.console_controller.export_console(filename)


    def _restore_logs(self):
        """
        Restores a message history in the console and displays it in the widget
        """
        filename, _ = QFileDialog.getOpenFileName(
            None, "Import console history", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return

        try:
            self._console_controller.import_console(filename)
            self._output.clear()

            for message in self._context.console.messages:
                self.print_message(message)
        except Exception:
            error_text = "Cannot open this file, incompatible format"
            error_modal = ErrorModal(self._main_window, error_text)
            error_modal.exec()

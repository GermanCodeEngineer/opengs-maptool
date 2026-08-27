from opengs_maptool.app import App
from opengs_maptool.ui.main_window import MainWindow
import asyncio, qasync

def main() -> int:
    # Import to initialize
    from opengs_maptool.services.logging_service import LOGGING_SERVICE

    app = App()
    
    # Initialize qasync event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow(app)
    window.show()

    with loop:
        return loop.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())

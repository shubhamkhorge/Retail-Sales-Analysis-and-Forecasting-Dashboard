# Simple script to launch the Streamlit dashboard

import subprocess
import sys
import os


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ["streamlit", "plotly"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install them using: pip install streamlit plotly")
        return False

    return True


def main():
    """Launch the Streamlit dashboard"""
    print("🚀 Starting Retail Sales Dashboard...")

    # Check dependencies
    if not check_dependencies():
        print("⚠️  Please install missing dependencies first")
        return

    # Check if dashboard file exists
    if not os.path.exists("dashboard.py"):
        print("❌ dashboard.py not found!")
        print("💡 Make sure dashboard.py is in the same directory")
        return

    try:
        # Launch Streamlit dashboard
        print("🌐 Launching dashboard at http://localhost:8501")
        print("📊 Dashboard will open in your web browser")
        print("⏹️  Press Ctrl+C to stop the dashboard")

        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped successfully!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Try running manually: streamlit run dashboard.py")


if __name__ == "__main__":
    main()

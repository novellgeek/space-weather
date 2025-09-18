import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class TacticalBox(QFrame):
    def __init__(self, label, value, color='#00ffe7'):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"background-color: {color}; border-radius: 10px;")
        layout = QVBoxLayout()
        lbl = QLabel(label)
        lbl.setFont(QFont("Arial", 12))  # Replace with Orbitron if installed
        lbl.setStyleSheet("color: #181a20;")
        val = QLabel(value)
        val.setFont(QFont("Arial", 18, QFont.Bold))  # Replace with Orbitron if installed
        val.setStyleSheet("color: #181a20;")
        layout.addWidget(lbl)
        layout.addWidget(val)
        self.setLayout(layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Space Weather Tactical Overview")
        self.setGeometry(200, 100, 900, 650)
        self.setStyleSheet("background-color: #181a20;")

        layout = QVBoxLayout()
        title = QLabel("Space Weather Tactical Overview")
        title.setFont(QFont("Arial", 20))  # Replace with Orbitron if available
        title.setStyleSheet("color: #00ffe7;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Chart placeholders
        chart_layout = QHBoxLayout()
        chart1 = QLabel("Solar X-ray Flux Chart\n[Plotly here]")
        chart1.setStyleSheet("color: #00ffe7; background-color: #222; border: 2px solid #00ffe7; border-radius: 8px;")
        chart2 = QLabel("Solar Proton Flux Chart\n[Plotly here]")
        chart2.setStyleSheet("color: #00ffe7; background-color: #222; border: 2px solid #00ffe7; border-radius: 8px;")
        chart3 = QLabel("Kp Index Chart\n[Plotly here]")
        chart3.setStyleSheet("color: #00ffe7; background-color: #222; border: 2px solid #00ffe7; border-radius: 8px;")
        chart_layout.addWidget(chart1)
        chart_layout.addWidget(chart2)
        chart_layout.addWidget(chart3)
        layout.addLayout(chart_layout)

        # Ratings and PDF button
        bottom_layout = QHBoxLayout()
        r_box = TacticalBox("R Rating", "R1", "#00ffe7")
        g_box = TacticalBox("G Rating", "G2", "#ff00cc")
        pdf_btn = QPushButton("Export PDF")
        pdf_btn.setStyleSheet("background-color: #00ffe7; color: #181a20; font-size: 16px; border-radius: 8px;")
        bottom_layout.addWidget(r_box)
        bottom_layout.addWidget(g_box)
        bottom_layout.addWidget(pdf_btn)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
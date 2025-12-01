"""
Report Generator for Vibration TCM System.
Generates HTML/Markdown reports from logged data.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "vibration_logs.db"


class ReportGenerator:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
    def generate_report(self, hours: int = 24) -> Dict:
        """Generate report for last N hours."""
        conn = sqlite3.connect(str(self.db_path))
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Query DL predictions
        query_dl = """
            SELECT * FROM predictions 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(
            query_dl, 
            conn, 
            params=(start_time.isoformat(), end_time.isoformat())
        )
        
        # Query ML predictions
        query_ml = """
            SELECT * FROM ml_predictions 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """
        
        df_ml = pd.read_sql_query(
            query_ml, 
            conn, 
            params=(start_time.isoformat(), end_time.isoformat())
        )
        
        conn.close()
        
        if df.empty:
            return self._empty_report(start_time, end_time)
        
        # Calculate statistics
        total_predictions = len(df)
        fault_count = df['is_fault'].sum()
        fault_rate = (fault_count / total_predictions * 100) if total_predictions > 0 else 0
        
        # Model agreement
        agreement_count = 0
        for _, row in df.iterrows():
            labels = [row['cnn1d_label'], row['lstm_label'], row['cnn2d_label']]
            if len(set(labels)) == 1:
                agreement_count += 1
        agreement_rate = (agreement_count / total_predictions * 100) if total_predictions > 0 else 0
        
        # Average confidences
        avg_cnn1d_conf = df['cnn1d_confidence'].mean()
        avg_lstm_conf = df['lstm_confidence'].mean()
        avg_cnn2d_conf = df['cnn2d_confidence'].mean()
        
        # ML stats
        ml_total = len(df_ml)
        ml_avg_conf = df_ml['confidence'].mean() if ml_total > 0 else 0
        ml_avg_time = df_ml['total_time_ms'].mean() if ml_total > 0 else 0
        
        # Merge ML predictions with DL predictions by timestamp
        # Create a dictionary for quick ML lookup
        ml_dict = {}
        if not df_ml.empty:
            for _, row in df_ml.iterrows():
                ml_dict[row['timestamp']] = row['label']
        
        # Recent faults with ML included
        recent_faults_df = df[df['is_fault'] == 1].head(10)[
            ['timestamp', 'cnn1d_label', 'lstm_label', 'cnn2d_label']
        ].copy()
        
        # Add ML label if available
        recent_faults_df['ml_label'] = recent_faults_df['timestamp'].apply(
            lambda ts: ml_dict.get(ts, 'N/A')
        )
        
        recent_faults = recent_faults_df.to_dict('records')
        
        return {
            'period': {
                'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'hours': hours
            },
            'summary': {
                'total_predictions': int(total_predictions),
                'fault_count': int(fault_count),
                'fault_rate': round(fault_rate, 2),
                'good_count': int(total_predictions - fault_count),
                'agreement_rate': round(agreement_rate, 2)
            },
            'models': {
                'cnn1d_avg_confidence': round(avg_cnn1d_conf, 3),
                'lstm_avg_confidence': round(avg_lstm_conf, 3),
                'cnn2d_avg_confidence': round(avg_cnn2d_conf, 3),
                'ml_total_predictions': int(ml_total),
                'ml_avg_confidence': round(ml_avg_conf, 3),
                'ml_avg_time_ms': round(ml_avg_time, 2)
            },
            'recent_faults': recent_faults
        }
        
    def _empty_report(self, start_time, end_time) -> Dict:
        """Return empty report structure."""
        return {
            'period': {
                'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'hours': 0
            },
            'summary': {
                'total_predictions': 0,
                'fault_count': 0,
                'fault_rate': 0.0,
                'good_count': 0,
                'agreement_rate': 0.0
            },
            'models': {
                'cnn1d_avg_confidence': 0.0,
                'lstm_avg_confidence': 0.0,
                'cnn2d_avg_confidence': 0.0,
                'ml_total_predictions': 0,
                'ml_avg_confidence': 0.0,
                'ml_avg_time_ms': 0.0
            },
            'recent_faults': []
        }
    
    def format_markdown(self, report: Dict) -> str:
        """Format report as Markdown."""
        md = f"""
# 🔧 Vibration TCM Daily Report

**Report Period**: {report['period']['start']} to {report['period']['end']}  
**Duration**: {report['period']['hours']} hours

---

## 📊 Summary Statistics

- **Total Predictions**: {report['summary']['total_predictions']}
- **Fault Detections**: {report['summary']['fault_count']} ({report['summary']['fault_rate']}%)
- **Good Detections**: {report['summary']['good_count']} ({100 - report['summary']['fault_rate']:.2f}%)
- **Model Agreement Rate**: {report['summary']['agreement_rate']}%

---

## 🤖 Model Performance

### Deep Learning Models
| Model | Avg Confidence |
|-------|----------------|
| CNN1D | {report['models']['cnn1d_avg_confidence']:.1%} |
| LSTM  | {report['models']['lstm_avg_confidence']:.1%} |
| CNN2D | {report['models']['cnn2d_avg_confidence']:.1%} |

### Machine Learning Model (RF-20)
- **Total Predictions**: {report['models']['ml_total_predictions']}
- **Avg Confidence**: {report['models']['ml_avg_confidence']:.1%}
- **Avg Inference Time**: {report['models']['ml_avg_time_ms']:.2f}ms

---

## ⚠️ Recent Fault Detections

"""
        if report['recent_faults']:
            for i, fault in enumerate(report['recent_faults'], 1):
                md += f"\n**{i}. {fault['timestamp']}**\n"
                md += f"  - CNN1D: {fault['cnn1d_label']}\n"
                md += f"  - LSTM: {fault['lstm_label']}\n"
                md += f"  - CNN2D: {fault['cnn2d_label']}\n"
                if 'ml_label' in fault and fault['ml_label'] != 'N/A':
                    md += f"  - ML (RF-20): {fault['ml_label']}\n"
        else:
            md += "\n✅ No faults detected during this period.\n"
        
        md += "\n---\n\n*Generated by Vibration TCM System*"
        
        return md


def main():
    parser = argparse.ArgumentParser(description="Generate Vibration TCM Report")
    parser.add_argument(
        "--hours", 
        type=int, 
        default=24, 
        help="Report period in hours (default: 24)"
    )
    parser.add_argument(
        "--db", 
        type=Path, 
        default=DB_PATH, 
        help="SQLite database path"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        help="Output file path (optional, prints to console if not specified)"
    )
    args = parser.parse_args()
    
    generator = ReportGenerator(args.db)
    report = generator.generate_report(args.hours)
    markdown = generator.format_markdown(report)
    
    if args.output:
        args.output.write_text(markdown, encoding='utf-8')
        print(f"Report saved to {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()

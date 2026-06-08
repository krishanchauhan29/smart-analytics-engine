import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdfplumber
import pytesseract
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import openpyxl
from io import BytesIO
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Smart Business Analytics Engine",
    page_icon="🚀",
    layout="wide"
)

# ==================== DATA LOADING ====================
def load_csv(file):
    raw = file.read()
    for encoding in ['utf-8', 'windows-1252', 'latin-1']:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            if df.shape[1] > 1:
                return df
        except:
            continue
    return None

def load_excel(file):
    return pd.read_excel(file)

def load_pdf(file):
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.extend(table)
    if tables:
        df = pd.DataFrame(tables[1:], columns=tables[0])
        return df
    else:
        st.warning("No tables found in PDF. Try CSV or Excel for better results.")
        return None

def load_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    lines = [line.split() for line in text.strip().split('\n') if line.strip()]
    if len(lines) > 1:
        df = pd.DataFrame(lines[1:], columns=lines[0])
        return df
    else:
        st.warning("Could not extract table from image. Try a clearer screenshot.")
        return None

def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return load_csv(uploaded_file)
    elif name.endswith(('.xlsx', '.xls')):
        return load_excel(uploaded_file)
    elif name.endswith('.pdf'):
        return load_pdf(uploaded_file)
    elif name.endswith(('.png', '.jpg', '.jpeg')):
        return load_image(uploaded_file)
    else:
        st.error("Unsupported format! Please upload CSV, Excel, PDF, or Image.")
        return None

# ==================== EDA FUNCTIONS ====================
def get_kpis(df):
    kpis = {
        'Total Rows': df.shape[0],
        'Total Columns': df.shape[1],
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Numeric Columns': len(df.select_dtypes(include=np.number).columns),
        'Categorical Columns': len(df.select_dtypes(include='object').columns)
    }
    return kpis

def get_insights(df):
    insights = []
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"⚠️ Dataset has {missing} missing values — consider imputation before modeling.")
    else:
        insights.append("✅ No missing values found — dataset is clean!")
    dups = df.duplicated().sum()
    if dups > 0:
        insights.append(f"⚠️ {dups} duplicate rows found — consider removing them.")
    else:
        insights.append("✅ No duplicate rows found.")
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].shape[0]
        if outliers > 0:
            insights.append(f"📊 Column '{col}' has {outliers} outliers detected via IQR method.")
    return insights

# ==================== ML FUNCTIONS ====================
def run_ml(df, target_col, task_type):
    df_ml = df.copy().dropna()
    le = LabelEncoder()
    for col in df_ml.select_dtypes(include='object').columns:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    X = df_ml.drop(columns=[target_col])
    y = df_ml[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task_type == 'Classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        metric_name = "Accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        metric_name = "R² Score"
    feat_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    return score, metric_name, feat_imp, y_test, y_pred

# ==================== EXCEL REPORT ====================
def generate_excel_report(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        df.describe().round(2).to_excel(writer, sheet_name='Summary Statistics')
        missing = df.isnull().sum().to_frame('Missing Values')
        missing['Missing %'] = (df.isnull().sum() / len(df) * 100).round(2)
        missing.to_excel(writer, sheet_name='Missing Values')
        dup_df = pd.DataFrame({
            'Metric': ['Total Rows', 'Duplicate Rows', 'Unique Rows'],
            'Count': [len(df), df.duplicated().sum(), len(df) - df.duplicated().sum()]
        })
        dup_df.to_excel(writer, sheet_name='Data Quality', index=False)
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            num_analysis = pd.DataFrame({
                'Column': num_cols,
                'Mean': df[num_cols].mean().round(2).values,
                'Median': df[num_cols].median().round(2).values,
                'Std Dev': df[num_cols].std().round(2).values,
                'Min': df[num_cols].min().values,
                'Max': df[num_cols].max().values
            })
            num_analysis.to_excel(writer, sheet_name='Numeric Analysis', index=False)
        cat_cols = df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            cat_analysis = pd.DataFrame({
                'Column': cat_cols,
                'Unique Values': df[cat_cols].nunique().values,
                'Most Frequent': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols],
                'Most Frequent Count': [df[col].value_counts().iloc[0] for col in cat_cols]
            })
            cat_analysis.to_excel(writer, sheet_name='Categorical Analysis', index=False)
    return output.getvalue()

# ==================== PDF REPORT ====================
def generate_pdf_report(df):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter,
                            rightMargin=inch*0.5, leftMargin=inch*0.5,
                            topMargin=inch*0.5, bottomMargin=inch*0.5)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=18, textColor=colors.HexColor('#2196F3'),
                                 spaceAfter=12)
    elements.append(Paragraph("Smart Business Analytics Report", title_style))
    elements.append(Paragraph("Generated by Krishan Kumar Chauhan | M.Tech Data Science, GBU",
                              styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("1. Dataset Overview", styles['Heading2']))
    overview_data = [
        ['Metric', 'Value'],
        ['Total Rows', str(df.shape[0])],
        ['Total Columns', str(df.shape[1])],
        ['Missing Values', str(df.isnull().sum().sum())],
        ['Duplicate Rows', str(df.duplicated().sum())],
        ['Numeric Columns', str(len(df.select_dtypes(include=np.number).columns))],
        ['Categorical Columns', str(len(df.select_dtypes(include='object').columns))],
    ]
    t = Table(overview_data, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2196F3')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("2. Summary Statistics (Numeric Columns)", styles['Heading2']))
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        stats = df[num_cols].describe().round(2)
        stat_data = [['Stat'] + list(num_cols)]
        for idx in stats.index:
            row = [idx] + [str(stats.loc[idx, col]) for col in num_cols]
            stat_data.append(row)
        col_width = 6.5*inch / len(stat_data[0])
        t2 = Table(stat_data, colWidths=[col_width]*len(stat_data[0]))
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("3. Missing Values Analysis", styles['Heading2']))
    missing_data = [['Column', 'Missing Count', 'Missing %']]
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = round(missing_count / len(df) * 100, 2)
        missing_data.append([col, str(missing_count), f"{missing_pct}%"])
    t3 = Table(missing_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#FF9800')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("4. Auto Business Insights", styles['Heading2']))
    insights = get_insights(df)
    for insight in insights:
        clean = insight.replace('✅','').replace('⚠️','').replace('📊','').strip()
        elements.append(Paragraph(f"• {clean}", styles['Normal']))

    doc.build(elements)
    return output.getvalue()

# ==================== MAIN UI ====================
st.title("🚀 Smart Business Analytics Engine")
st.markdown("**Upload any business dataset and get instant EDA, visualizations, ML predictions, and downloadable reports!**")
st.markdown("---")

st.sidebar.image("https://img.icons8.com/color/96/analytics.png", width=80)
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")
st.sidebar.info("""
**Supported Formats:**
- 📊 CSV (Best)
- 📗 Excel (.xlsx, .xls)
- 📄 PDF (digital tables)
- 🖼️ Image (clear screenshots)

For best results, upload a CSV or Excel file. PDF files work well if they contain digital tables (not scanned). Images work too, though handwritten or low-quality scans may not read perfectly — a clear screenshot works best!
""")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload your dataset",
    type=['csv', 'xlsx', 'xls', 'pdf', 'png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    with st.spinner("Loading your data..."):
        df = load_data(uploaded_file)

    if df is not None:
        df.columns = df.columns.str.strip()
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        st.success(f"✅ Data loaded successfully! {df.shape[0]} rows × {df.shape[1]} columns")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Overview", "📊 EDA", "📈 Visualizations", "🤖 ML Prediction", "📥 Download Report"
        ])

        with tab1:
            st.subheader("📋 Dataset Overview")
            kpis = get_kpis(df)
            cols = st.columns(6)
            icons = ['🗂️', '📊', '❓', '🔁', '🔢', '🔤']
            for i, (key, val) in enumerate(kpis.items()):
                cols[i].metric(f"{icons[i]} {key}", val)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔍 First 5 Rows")
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                st.subheader("📊 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Missing': df.isnull().sum().values,
                    'Unique': df.nunique().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            st.markdown("---")
            st.subheader("💡 Auto Business Insights")
            insights = get_insights(df)
            for insight in insights:
                st.markdown(f"- {insight}")

        with tab2:
            st.subheader("📊 Exploratory Data Analysis")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            if num_cols:
                st.markdown("#### 📈 Statistical Summary")
                st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
                st.markdown("#### 🔥 Correlation Heatmap")
                if len(num_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                               cmap='coolwarm', ax=ax, linewidths=0.5)
                    st.pyplot(fig)
                st.markdown("#### 📦 Outlier Detection (Boxplots)")
                selected_col = st.selectbox("Select column for boxplot", num_cols)
                fig = px.box(df, y=selected_col, title=f'Boxplot — {selected_col}',
                            color_discrete_sequence=['#2196F3'])
                st.plotly_chart(fig, use_container_width=True)
            if cat_cols:
                st.markdown("#### 🏷️ Categorical Columns Distribution")
                selected_cat = st.selectbox("Select categorical column", cat_cols)
                val_counts = df[selected_cat].value_counts().head(15)
                fig = px.bar(x=val_counts.index, y=val_counts.values,
                            title=f'Distribution — {selected_cat}',
                            color=val_counts.values,
                            color_continuous_scale='Blues',
                            labels={'x': selected_cat, 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("📈 Interactive Visualizations")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            all_cols = df.columns.tolist()
            chart_type = st.selectbox("Select Chart Type",
                                      ["Bar Chart", "Line Chart", "Scatter Plot",
                                       "Histogram", "Pie Chart", "Heatmap"])
            if chart_type == "Bar Chart":
                x = st.selectbox("X axis", all_cols, key='bar_x')
                y = st.selectbox("Y axis", num_cols, key='bar_y')
                fig = px.bar(df, x=x, y=y, title=f'{y} by {x}',
                            color_discrete_sequence=['#2196F3'])
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Line Chart":
                x = st.selectbox("X axis", all_cols, key='line_x')
                y = st.selectbox("Y axis", num_cols, key='line_y')
                fig = px.line(df, x=x, y=y, title=f'{y} over {x}',
                             color_discrete_sequence=['#E53935'])
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Scatter Plot":
                x = st.selectbox("X axis", num_cols, key='sc_x')
                y = st.selectbox("Y axis", num_cols, key='sc_y')
                color = st.selectbox("Color by (optional)", ['None'] + all_cols)
                fig = px.scatter(df, x=x, y=y,
                                color=None if color == 'None' else color,
                                title=f'{x} vs {y}')
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Histogram":
                col = st.selectbox("Select column", num_cols, key='hist_col')
                bins = st.slider("Number of bins", 10, 100, 30)
                fig = px.histogram(df, x=col, nbins=bins,
                                  title=f'Distribution of {col}',
                                  color_discrete_sequence=['#4CAF50'])
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Pie Chart":
                col = st.selectbox("Select column", all_cols, key='pie_col')
                val_counts = df[col].value_counts().head(10)
                fig = px.pie(values=val_counts.values, names=val_counts.index,
                            title=f'Distribution of {col}', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Heatmap":
                num_cols_heat = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols_heat) > 1:
                    fig = px.imshow(df[num_cols_heat].corr().round(2),
                                   title='Correlation Heatmap',
                                   color_continuous_scale='RdBu_r',
                                   text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("🤖 ML Prediction")
            st.markdown("Select a target column and let the engine train a model automatically!")
            target_col = st.selectbox("🎯 Select Target Column", df.columns.tolist())
            task_type = st.radio("Task Type", ["Classification", "Regression"])
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    try:
                        score, metric_name, feat_imp, y_test, y_pred = run_ml(df, target_col, task_type)
                        st.success("✅ Model trained successfully!")
                        st.metric(f"🏆 {metric_name}", f"{score*100:.2f}%" if metric_name == "Accuracy" else f"{score:.4f}")
                        st.markdown("#### 🔑 Top 10 Feature Importances")
                        fig = px.bar(feat_imp, x='Importance', y='Feature',
                                    orientation='h', title='Feature Importance',
                                    color='Importance',
                                    color_continuous_scale='Blues')
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("#### 📊 Actual vs Predicted (Sample)")
                        comparison = pd.DataFrame({
                            'Actual': list(y_test)[:20],
                            'Predicted': list(y_pred)[:20]
                        })
                        st.dataframe(comparison, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)} — Make sure target column has enough data!")

        with tab5:
            st.subheader("📥 Download Report")
            st.markdown("Download your complete analysis as Excel or PDF!")
            st.markdown("**Report includes:**")
            st.markdown("- ✅ Raw Data")
            st.markdown("- ✅ Summary Statistics")
            st.markdown("- ✅ Missing Values Analysis")
            st.markdown("- ✅ Data Quality Check")
            st.markdown("- ✅ Numeric & Categorical Analysis")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                excel_data = generate_excel_report(df)
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name="analytics_report.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            with col2:
                pdf_data = generate_pdf_report(df)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name="analytics_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

else:
    st.markdown("## 👋 Welcome to Smart Business Analytics Engine!")
    st.markdown("Upload any dataset from the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📋 **Auto EDA**\nAutomatic exploratory data analysis with KPIs, missing values, outliers")
    with col2:
        st.info("📈 **Interactive Charts**\nBar, Line, Scatter, Histogram, Pie, Heatmap — you choose!")
    with col3:
        st.info("🤖 **ML Prediction**\nSelect any column as target — model trains automatically!")

st.markdown("---")
st.caption("Built by Krishan Kumar Chauhan | M.Tech Data Science, GBU | Smart Analytics Engine v1.0")
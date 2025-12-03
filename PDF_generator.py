"""
Automated Insight Engine - Netflix Analytics
Streamlit App with PDF Export & Gemini AI
Author: Senior Data Scientist

Dataset: Netflix Movies and TV Shows (Small, Easy to Download)
Kaggle: https://www.kaggle.com/datasets/shivamb/netflix-shows

Time to Build: 3 Hours
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from pathlib import Path

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfgen import canvas

# AI Integration
import google.generativeai as genai
import json

# ============================================================================
# PAGE CONFIG - Make it Beautiful!
# ============================================================================

st.set_page_config(
    page_title="Netflix Analytics Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Beautiful UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ff88;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff;
        font-size: 16px;
    }
    
    /* Headers */
    h1 {
        color: #ff0000;
        text-align: center;
        font-size: 3rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        padding: 20px;
        background: linear-gradient(90deg, #ff0000, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2 {
        color: #00ff88;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    h3 {
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Cards */
    .css-1r6slb0 {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff0000, #ff6b6b);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 0, 0, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 2px dashed #00ff88;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        color: #00ff88 !important;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
    }
    
    .stInfo {
        background-color: rgba(0, 174, 255, 0.1);
        border-left: 4px solid #00aeff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # User will replace this
    MODEL_NAME = "gemini-1.5-flash"
    DATASET_URL = "https://www.kaggle.com/datasets/shivamb/netflix-shows"

# ============================================================================
# DATA PROCESSING
# ============================================================================

@st.cache_data
def load_data(file):
    """Load and preprocess Netflix data"""
    df = pd.read_csv(file)
    
    # Data cleaning
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    # Handle missing values
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Not Rated')
    
    # Extract first country
    df['primary_country'] = df['country'].str.split(',').str[0].str.strip()
    
    return df

def generate_analytics(df):
    """Generate comprehensive analytics"""
    
    analytics = {
        'total_titles': len(df),
        'movies': len(df[df['type'] == 'Movie']),
        'tv_shows': len(df[df['type'] == 'TV Show']),
        'countries': df['primary_country'].nunique(),
        'avg_release_year': df['release_year'].mean(),
        'latest_additions': df['date_added'].max(),
        
        # Time series
        'yearly_additions': df.groupby('year_added').size().to_dict(),
        'monthly_pattern': df.groupby('month_added').size().to_dict(),
        
        # Top statistics
        'top_countries': df['primary_country'].value_counts().head(10).to_dict(),
        'top_ratings': df['rating'].value_counts().head(10).to_dict(),
        'top_genres': df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10).to_dict(),
        
        # Type distribution
        'type_distribution': df['type'].value_counts().to_dict(),
        
        # Recent trends
        'recent_year_growth': calculate_growth_rate(df)
    }
    
    return analytics

def calculate_growth_rate(df):
    """Calculate year-over-year growth"""
    yearly = df.groupby('year_added').size().sort_index()
    if len(yearly) >= 2:
        recent = yearly.iloc[-2:]
        if recent.iloc[0] > 0:
            growth = ((recent.iloc[1] - recent.iloc[0]) / recent.iloc[0]) * 100
            return round(growth, 2)
    return 0

# ============================================================================
# AI INSIGHTS GENERATION
# ============================================================================

def initialize_gemini(api_key):
    """Initialize Gemini AI"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(Config.MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

def generate_executive_summary(model, analytics):
    """Generate executive summary using Gemini"""
    prompt = f"""
    As a senior data analyst, write a concise 3-sentence executive summary for Netflix content analytics:
    
    Data Summary:
    - Total Titles: {analytics['total_titles']}
    - Movies: {analytics['movies']}, TV Shows: {analytics['tv_shows']}
    - Countries: {analytics['countries']}
    - Recent Growth Rate: {analytics['recent_year_growth']}%
    - Top Country: {list(analytics['top_countries'].keys())[0]}
    
    Focus on: overall scale, content mix, and growth trajectory.
    Be specific with numbers and actionable.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Netflix has {analytics['total_titles']} titles with {analytics['recent_year_growth']}% recent growth."

def generate_ai_insights(model, analytics):
    """Generate specific insights using Gemini"""
    prompt = f"""
    Analyze this Netflix content data and generate exactly 4 specific insights in JSON format.
    
    Data:
    - Total: {analytics['total_titles']} titles
    - Movies: {analytics['movies']}, TV Shows: {analytics['tv_shows']}
    - Growth: {analytics['recent_year_growth']}%
    - Top Countries: {list(analytics['top_countries'].keys())[:3]}
    - Top Genres: {list(analytics['top_genres'].keys())[:3]}
    
    Return ONLY a JSON array with this exact structure:
    [
        {{"title": "Short Title", "insight": "Specific insight with numbers", "type": "success"}},
        {{"title": "Another Title", "insight": "Another insight", "type": "warning"}}
    ]
    
    Types: "success" for positive findings, "warning" for opportunities.
    Make insights specific and data-driven.
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean JSON from markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        insights = json.loads(text)
        return insights[:4]  # Ensure only 4 insights
    except Exception as e:
        # Fallback insights
        return [
            {"title": "Content Scale", "insight": f"Netflix has {analytics['total_titles']} titles available.", "type": "success"},
            {"title": "Content Mix", "insight": f"Movies dominate with {analytics['movies']} titles vs {analytics['tv_shows']} TV shows.", "type": "success"},
            {"title": "Growth Trend", "insight": f"Recent growth rate of {analytics['recent_year_growth']}% indicates strong expansion.", "type": "success"},
            {"title": "Global Reach", "insight": f"Content spans {analytics['countries']} countries showing international presence.", "type": "success"}
        ]

def generate_recommendations(model, insights):
    """Generate actionable recommendations"""
    prompt = f"""
    Based on these Netflix insights, generate exactly 4 specific, actionable recommendations.
    
    Insights:
    {json.dumps(insights, indent=2)}
    
    Return ONLY a JSON array of strings:
    ["Recommendation 1", "Recommendation 2", "Recommendation 3", "Recommendation 4"]
    
    Each should be specific, measurable, and actionable (1-2 sentences).
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        recommendations = json.loads(text)
        return recommendations[:4]
    except Exception as e:
        return [
            "Expand content production in high-growth markets",
            "Invest in original TV series to balance content portfolio",
            "Develop localized content for underserved regions",
            "Optimize content acquisition based on viewer preferences"
        ]

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_yearly_trend_chart(df):
    """Create yearly additions trend"""
    yearly = df.groupby('year_added').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly['year_added'],
        y=yearly['count'],
        mode='lines+markers',
        line=dict(color='#ff0000', width=3),
        marker=dict(size=10, color='#ff6b6b'),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',
        name='Additions'
    ))
    
    fig.update_layout(
        title='Netflix Content Additions Over Time',
        xaxis_title='Year',
        yaxis_title='Number of Titles',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_content_type_chart(df):
    """Create content type distribution"""
    type_counts = df['type'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.4,
        marker=dict(colors=['#ff0000', '#00ff88']),
        textinfo='label+percent',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title='Content Type Distribution',
        template='plotly_dark',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_top_countries_chart(df):
    """Create top countries bar chart"""
    top_countries = df['primary_country'].value_counts().head(10)
    
    fig = go.Figure(data=[go.Bar(
        x=top_countries.values,
        y=top_countries.index,
        orientation='h',
        marker=dict(
            color=top_countries.values,
            colorscale='Reds',
            showscale=True
        ),
        text=top_countries.values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Top 10 Countries by Content',
        xaxis_title='Number of Titles',
        yaxis_title='Country',
        template='plotly_dark',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_genre_chart(df):
    """Create top genres chart"""
    genres = df['listed_in'].str.split(',').explode().str.strip()
    top_genres = genres.value_counts().head(10)
    
    fig = go.Figure(data=[go.Bar(
        x=top_genres.index,
        y=top_genres.values,
        marker=dict(
            color=top_genres.values,
            colorscale='Viridis',
            showscale=True
        ),
        text=top_genres.values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Top 10 Genres',
        xaxis_title='Genre',
        yaxis_title='Number of Titles',
        template='plotly_dark',
        height=400,
        xaxis={'tickangle': -45},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# ============================================================================
# PDF GENERATION
# ============================================================================

def save_plotly_as_image(fig, filename):
    """Save plotly figure as image"""
    img_bytes = fig.to_image(format="png", width=800, height=400)
    with open(filename, 'wb') as f:
        f.write(img_bytes)
    return filename

def generate_pdf_report(df, analytics, insights, recommendations, summary):
    """Generate comprehensive PDF report"""
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#E50914'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#E50914'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Netflix Analytics Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Automated Insight Engine", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(summary, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Key Metrics Table
    story.append(Paragraph("Key Performance Indicators", heading_style))
    
    kpi_data = [
        ['Metric', 'Value'],
        ['Total Titles', f"{analytics['total_titles']:,}"],
        ['Movies', f"{analytics['movies']:,}"],
        ['TV Shows', f"{analytics['tv_shows']:,}"],
        ['Countries', f"{analytics['countries']:,}"],
        ['Growth Rate', f"{analytics['recent_year_growth']}%"],
    ]
    
    kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E50914')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(kpi_table)
    story.append(Spacer(1, 0.5*inch))
    
    # AI Insights
    story.append(Paragraph("AI-Powered Insights", heading_style))
    
    for idx, insight in enumerate(insights, 1):
        title = f"<b>{idx}. {insight['title']}</b>"
        story.append(Paragraph(title, styles['Normal']))
        story.append(Paragraph(insight['insight'], styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Strategic Recommendations", heading_style))
    
    for idx, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{idx}.</b> {rec}", styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown("<h1>🎬 NETFLIX ANALYTICS ENGINE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00ff88; font-size: 1.2rem;'>Powered by AI • Automated Insights • PDF Export</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
        st.markdown("---")
        
        st.markdown("### 🔑 Configuration")
        api_key = st.text_input("Gemini API Key", type="password", value="", 
                                help="Get your API key from Google AI Studio")
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info("""
        **Netflix Movies & TV Shows**
        
        Download from Kaggle:
        [Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
        
        File: `netflix_titles.csv`
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Enter your Gemini API key
        2. Upload the Netflix CSV
        3. Click 'Generate Report'
        4. Download the PDF
        """)
    
    # File Upload
    st.markdown("### 📁 Upload Netflix Dataset")
    uploaded_file = st.file_uploader("Choose netflix_titles.csv", type=['csv'], 
                                     help="Upload the Netflix dataset from Kaggle")
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("🔄 Loading and processing data..."):
            df = load_data(uploaded_file)
            analytics = generate_analytics(df)
        
        st.success(f"✅ Loaded {len(df):,} Netflix titles successfully!")
        
        # Quick Stats
        st.markdown("### 📈 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Titles", f"{analytics['total_titles']:,}", 
                     delta=f"{analytics['recent_year_growth']}% growth")
        with col2:
            st.metric("Movies", f"{analytics['movies']:,}")
        with col3:
            st.metric("TV Shows", f"{analytics['tv_shows']:,}")
        with col4:
            st.metric("Countries", f"{analytics['countries']:,}")
        
        # Generate Report Button
        st.markdown("---")
        if st.button("🚀 Generate AI-Powered Report", use_container_width=True):
            if not api_key:
                st.error("⚠️ Please enter your Gemini API key in the sidebar!")
            else:
                with st.spinner("🤖 AI is analyzing your data..."):
                    # Initialize Gemini
                    model = initialize_gemini(api_key)
                    
                    if model:
                        # Generate AI content
                        summary = generate_executive_summary(model, analytics)
                        insights = generate_ai_insights(model, analytics)
                        recommendations = generate_recommendations(model, insights)
                        
                        # Store in session state
                        st.session_state.summary = summary
                        st.session_state.insights = insights
                        st.session_state.recommendations = recommendations
                        st.session_state.report_generated = True
                        
                        st.success("✅ Report generated successfully!")
                        st.balloons()
        
        # Display Report if generated
        if st.session_state.get('report_generated', False):
            st.markdown("---")
            
            # Executive Summary
            st.markdown("### 📋 Executive Summary")
            st.info(st.session_state.summary)
            
            # Visualizations
            st.markdown("### 📊 Visual Analytics")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🎭 Content Type", "🌍 Countries", "🎬 Genres"])
            
            with tab1:
                fig1 = create_yearly_trend_chart(df)
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = create_content_type_chart(df)
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                fig3 = create_top_countries_chart(df)
                st.plotly_chart(fig3, use_container_width=True)
            
            with tab4:
                fig4 = create_genre_chart(df)
                st.plotly_chart(fig4, use_container_width=True)
            
            # AI Insights
            st.markdown("### 🤖 AI-Powered Insights")
            
            col1, col2 = st.columns(2)
            
            for idx, insight in enumerate(st.session_state.insights):
                with col1 if idx % 2 == 0 else col2:
                    icon = "✅" if insight['type'] == 'success' else "⚠️"
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; 
                                border-left: 4px solid {"#00ff88" if insight["type"] == "success" else "#ffa500"}; 
                                margin-bottom: 10px;'>
                        <h4 style='color: #00ff88; margin: 0;'>{icon} {insight['title']}</h4>
                        <p style='color: #ffffff; margin-top: 10px;'>{insight['insight']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Strategic Recommendations")
            
            for idx, rec in enumerate(st.session_state.recommendations, 1):
                st.markdown(f"""
                <div style='background: rgba(0,255,136,0.1); padding: 15px; border-radius: 8px; 
                            margin-bottom: 10px; border-left: 4px solid #00ff88;'>
                    <strong style='color: #00ff88;'>{idx}.</strong> 
                    <span style='color: #ffffff;'>{rec}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # PDF Export
            st.markdown("---")
            st.markdown("### 📄 Export Report")
            
            if st.button("📥 Generate PDF Report", use_container_width=True):
                with st.spinner("🔄 Generating PDF..."):
                    pdf_buffer = generate_pdf_report(
                        df, analytics, 
                        st.session_state.insights, 
                        st.session_state.recommendations,
                        st.session_state.summary
                    )
                    
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"Netflix_Analytics_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF ready for download!")
    
    else:
        # Landing page
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #00ff88;'>Welcome to Netflix Analytics Engine!</h2>
            <p style='color: #ffffff; font-size: 1.1rem;'>Upload your Netflix dataset to get started</p>
            <br>
            <p style='color: #aaaaaa;'>
                This tool analyzes Netflix content data and generates:<br>
                ✨ AI-powered insights using Google Gemini<br>
                📊 Interactive visualizations<br>
                📄 Professional PDF reports<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with ❤️ by Senior Data Scientist | Powered by Streamlit & Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
"""
Run these commands to install dependencies:

pip install streamlit pandas numpy plotly google-generativeai reportlab kaleido

To run the app:
streamlit run app.py

Dataset Download:
1. Go to https://www.kaggle.com/datasets/shivamb/netflix-shows
2. Download netflix_titles.csv
3. Upload it in the Streamlit app

Gemini API Key:
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Paste it in the sidebar

The app will:
- Analyze the Netflix dataset
- Generate AI insights using Gemini
- Create beautiful visualizations
- Export a professional PDF report
"""
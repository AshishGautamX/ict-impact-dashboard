# ICT Impact Assessment Dashboard

A comprehensive AI-powered web application for assessing the impact of Information & Communication Technologies (ICT) on medical college libraries in Bihar. This platform provides data collection, analysis, and AI-powered predictions to help improve library services and infrastructure.

## 🌟 Overview

The ICT Impact Assessment Dashboard is a full-stack web application designed to:
- Collect comprehensive data about ICT infrastructure and usage in medical college libraries
- Analyze patterns and trends in library services and user satisfaction
- Provide AI-powered predictions and recommendations for improvements
- Generate detailed reports and visualizations for decision-making
- Enable administrators to manage users and monitor system performance

## 🚀 Key Features

### 📊 **Data Collection & Management**
- **Multi-step Questionnaire**: Comprehensive data collection with progress tracking
- **Auto-save Functionality**: Prevents data loss during form completion
- **Data Validation**: Real-time validation with helpful error messages
- **Bulk Import**: CSV/Excel import with data validation and error reporting
- **Entry Management**: Admin approval workflow for data quality control

### 🤖 **AI-Powered Analytics**
- **Satisfaction Prediction**: ML models predict user satisfaction levels
- **Efficiency Analysis**: Automated assessment of service efficiency
- **Scenario Simulation**: What-if analysis for proposed improvements
- **Smart Recommendations**: AI-generated suggestions for infrastructure improvements
- **College Clustering**: Automatic grouping of similar institutions

### 📈 **Interactive Visualizations**
- **Real-time Dashboards**: Live updating charts and statistics
- **Infrastructure Analysis**: Detailed breakdown of ICT components
- **Satisfaction Trends**: Visual analysis of user satisfaction patterns
- **Barrier Analysis**: Identification and visualization of implementation barriers
- **Correlation Matrix**: Statistical relationships between variables

### 👥 **User Management**
- **Role-based Access**: Admin and regular user roles with different permissions
- **User Authentication**: Secure login/logout with JWT tokens
- **Profile Management**: User profile creation and management
- **Activity Tracking**: Monitor user activities and system usage

### 📋 **Report Generation**
- **Automated Reports**: Generate comprehensive analysis reports
- **Multiple Formats**: Export data in CSV, Excel, and PDF formats
- **Custom Filtering**: Filter data by college, date range, or other criteria
- **Statistical Summaries**: Automated calculation of key metrics

## 🛠️ Technology Stack

### **Frontend**
- **React 18** with TypeScript for type-safe development
- **Vite** for fast development and building
- **Tailwind CSS** for responsive and modern UI design
- **Recharts** for interactive data visualizations
- **React Query** for efficient data fetching and caching
- **Zustand** for lightweight state management
- **React Hook Form** for form handling and validation
- **Framer Motion** for smooth animations

### **Backend**
- **FastAPI** (Python 3.10+) for high-performance API development
- **SQLAlchemy** for database ORM and migrations
- **Pydantic** for data validation and serialization
- **Scikit-learn** for machine learning models
- **Pandas & NumPy** for data processing and analysis
- **JWT** for secure authentication
- **CORS** middleware for cross-origin requests

### **Database**
- **SQLite** for development (included)
- **PostgreSQL** support for production deployments
- **Automated migrations** with Alembic

### **Machine Learning**
- **7 Trained ML Models**:
  - Satisfaction Classifier (XGBoost)
  - Efficiency Regressor (XGBoost)
  - College Clustering (K-Means)
  - ROI Predictor
  - Enhanced Efficiency Ensemble (Random Forest + Gradient Boosting + Neural Network)
  - Scenario Impact Simulator
  - AI Recommendation Engine

## 📋 Prerequisites

Before running the application, ensure you have:

- **Node.js 18+** and **npm** installed
- **Python 3.10+** installed
- **Git** for cloning the repository
- **Code editor** (VS Code recommended)

## 🚀 Installation & Setup

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd ict-impact-dashboard
```

### 2. **Backend Setup**

#### Install Python Dependencies
```bash
cd backend
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### Initialize Database
```bash
python init_database.py
```

#### Start Backend Server
```bash
# Using the provided script (Windows)
powershell -ExecutionPolicy Bypass -File start-backend.ps1

# Or manually
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`

### 3. **Frontend Setup**

#### Install Node Dependencies
```bash
cd frontend
npm install
```

#### Start Development Server
```bash
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### 4. **Access the Application**

1. Open your browser and navigate to `http://localhost:5173`
2. Create an account or use the default admin credentials:
   - **Email**: `admin@example.com`
   - **Password**: `admin123`

## 🎯 How to Use the Application

### **For Regular Users**

1. **Registration & Login**
   - Create an account with your email and password
   - Login to access the dashboard

2. **Data Entry**
   - Navigate to "Data Entry" from the sidebar
   - Fill out the comprehensive questionnaire about your library
   - The form auto-saves progress and validates input
   - Submit when complete

3. **View Analytics**
   - Visit "Analysis" to see data visualizations
   - Explore infrastructure, satisfaction, and barrier analysis
   - View correlation matrices and trends

4. **AI Predictions**
   - Go to "Prediction Lab" to use AI features
   - Get satisfaction predictions based on your data
   - Analyze efficiency scores and improvement potential
   - Run scenario simulations for proposed changes
   - Receive AI-powered recommendations

5. **Generate Reports**
   - Access "Reports" to create custom reports
   - Filter data by various criteria
   - Export reports in multiple formats

### **For Administrators**

1. **Admin Panel Access**
   - Login with admin credentials
   - Access "Admin Panel" from the sidebar

2. **User Management**
   - View all registered users
   - Activate/deactivate user accounts
   - Delete users if necessary
   - Monitor user activity and response counts

3. **Entry Management**
   - Review and approve submitted entries
   - Reject entries with reasons if needed
   - Delete inappropriate or duplicate entries
   - Monitor data quality

4. **AI Model Management**
   - View training status and metrics
   - Retrain models when new data is available
   - Monitor model performance
   - Force retrain if needed

5. **System Monitoring**
   - View system statistics and usage
   - Monitor database size and performance
   - Track training history and metrics

## 🤖 Machine Learning Models

The application includes 7 trained ML models:

### **Core Models**
1. **Satisfaction Classifier**: Predicts user satisfaction levels (High/Medium/Low)
2. **Efficiency Regressor**: Estimates service efficiency scores (1-10 scale)
3. **College Clusterer**: Groups similar institutions for comparison

### **Enhanced Models**
4. **Enhanced Efficiency Ensemble**: Combines multiple algorithms for better accuracy
5. **Scenario Impact Simulator**: Predicts outcomes of proposed changes
6. **AI Recommendation Engine**: Generates intelligent improvement suggestions
7. **ROI Predictor**: Estimates return on investment for improvements

### **Model Features**
- **Automatic Training**: Models retrain automatically with new data
- **Feature Importance**: Shows which factors most influence predictions
- **Confidence Intervals**: Provides uncertainty estimates
- **Cross-validation**: Ensures model reliability

## 📁 Project Structure

```
ict-impact-dashboard/
├── frontend/                          # React frontend application
│   ├── src/
│   │   ├── components/               # Reusable UI components
│   │   │   ├── ui/                  # Basic UI components
│   │   │   ├── layout/              # Layout components (Navbar, Sidebar)
│   │   │   ├── forms/               # Form components
│   │   │   ├── charts/              # Chart components
│   │   │   ├── ai/                  # AI-related components
│   │   │   └── admin/               # Admin-specific components
│   │   ├── pages/                   # Page components
│   │   │   ├── Home.tsx             # Dashboard home page
│   │   │   ├── DataEntry.tsx        # Data entry form
│   │   │   ├── Analysis.tsx         # Analytics dashboard
│   │   │   ├── PredictionLab.tsx    # AI predictions interface
│   │   │   ├── Reports.tsx          # Report generation
│   │   │   ├── Admin.tsx            # Admin panel
│   │   │   ├── Login.tsx            # Login page
│   │   │   └── Signup.tsx           # Registration page
│   │   ├── lib/                     # Utilities and configurations
│   │   │   ├── api.ts               # API client
│   │   │   └── utils.ts             # Helper functions
│   │   ├── store/                   # State management
│   │   │   └── store.ts             # Zustand store
│   │   ├── types/                   # TypeScript type definitions
│   │   └── App.tsx                  # Main application component
│   ├── package.json                 # Frontend dependencies
│   └── vite.config.ts              # Vite configuration
├── backend/                         # FastAPI backend application
│   ├── app/
│   │   ├── main.py                  # FastAPI application entry point
│   │   ├── models/                  # Data models
│   │   │   ├── db_models.py         # SQLAlchemy database models
│   │   │   └── schemas.py           # Pydantic schemas
│   │   ├── routes/                  # API route handlers
│   │   │   ├── auth.py              # Authentication routes
│   │   │   ├── data.py              # Data management routes
│   │   │   ├── predictions.py       # AI prediction routes
│   │   │   ├── analysis.py          # Analytics routes
│   │   │   ├── reports.py           # Report generation routes
│   │   │   └── admin.py             # Admin routes
│   │   ├── services/                # Business logic services
│   │   │   ├── auth_service.py      # Authentication service
│   │   │   ├── data_service.py      # Data management service
│   │   │   ├── ml_service.py        # Machine learning service
│   │   │   ├── training_service.py  # Model training service
│   │   │   └── db_data_service.py   # Database operations
│   │   ├── utils/                   # Utility functions
│   │   │   ├── database_seeder.py   # Database seeding
│   │   │   └── init_database.py     # Database initialization
│   │   └── database.py              # Database configuration
│   ├── database/                    # Database files
│   │   └── ict_survey.db           # SQLite database (auto-generated)
│   ├── ml_models/                   # Machine learning models
│   │   └── ict_ml_models_complete.pkl  # Trained models (auto-generated)
│   ├── requirements.txt             # Python dependencies
│   ├── init_database.py            # Database initialization script
│   ├── start-backend.ps1           # Windows startup script
│   └── Dockerfile                  # Docker configuration
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```


## 🧰 Developer / Troubleshooting Notes

If you see "failed to fetch" in the browser when attempting to log in, follow these quick checks:

- Is the backend running? Check the health endpoint:

```powershell
Invoke-RestMethod 'http://localhost:8000/health'
# should return: {"status":"healthy"}
```

- Make sure the frontend and backend origins match CORS settings. By default the backend allows:
   - http://localhost:5173
   - http://127.0.0.1:5173
   - http://localhost:3000

   If you run the frontend on a different port, set the `CORS_ORIGINS` environment variable (comma-separated) before starting the backend:

```powershell
# Windows (PowerShell)
$env:CORS_ORIGINS = 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000'
# then start backend
powershell -ExecutionPolicy Bypass -File backend\start-backend.ps1
```

- The frontend uses `VITE_API_URL` to talk to the API. Set it in the frontend dev environment if your backend runs on a different host/port:

Create a `.env.local` inside `frontend/` with:

```
VITE_API_URL=http://127.0.0.1:8000
```

- Report generation imports `matplotlib` and `seaborn`. To avoid startup failures the project now imports those libraries lazily when a report is generated. If you need PDF/Excel charts, install the plotting dependencies in the backend venv:

```powershell
# activate venv
& backend\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

If you'd rather not install plotting libs for development, the report generation functions will raise an ImportError only when invoked (and not on server start).

If you'd like, I can also:
- Add `seaborn` to `backend/requirements.txt` (done),
- Convert the reports route to import the report generator only when the reports endpoints are called, or
- Add a small troubleshooting script that checks backend availability and CORS from your dev machine.

## 🔧 Configuration

### **Environment Variables**

Create `.env` files for configuration:

**Frontend (.env)**
```env
VITE_API_URL=http://localhost:8000
```

**Backend (.env)** (optional)
```env
DATABASE_URL=sqlite:///./database/ict_survey.db
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### **Database Configuration**

The application uses SQLite by default, which requires no additional setup. For production, you can configure PostgreSQL:

```env
DATABASE_URL=postgresql://username:password@localhost/dbname
```

## 📊 API Documentation

### **Authentication Endpoints**
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info

### **Data Management Endpoints**
- `POST /api/data/submit` - Submit survey response
- `GET /api/data/all` - Get all responses (with filtering)
- `GET /api/data/summary` - Get summary statistics
- `PUT /api/data/update/{id}` - Update response
- `DELETE /api/data/delete/{id}` - Delete response
- `POST /api/data/bulk-import` - Bulk import from CSV/Excel

### **AI Prediction Endpoints**
- `POST /api/predict/satisfaction` - Predict satisfaction level
- `POST /api/predict/efficiency` - Predict efficiency score
- `POST /api/predict/scenario` - Simulate improvement scenarios
- `GET /api/predict/clusters` - Get college clusters
- `GET /api/predict/recommendations` - Get AI recommendations

### **Analytics Endpoints**
- `GET /api/analysis/infrastructure` - Infrastructure analysis
- `GET /api/analysis/satisfaction` - Satisfaction analysis
- `GET /api/analysis/barriers` - Barrier analysis
- `GET /api/analysis/correlation` - Correlation matrix

### **Admin Endpoints**
- `GET /api/admin/users` - Get all users
- `PUT /api/admin/users/{id}/status` - Update user status
- `DELETE /api/admin/users/{id}` - Delete user
- `GET /api/admin/entries` - Get all entries for review
- `PUT /api/admin/entries/{id}/approve` - Approve entry
- `PUT /api/admin/entries/{id}/reject` - Reject entry
- `POST /api/admin/training/retrain` - Retrain ML models
- `GET /api/admin/training/status` - Get training status

## 🚀 Deployment

### **Frontend Deployment (Vercel/Netlify)**

1. Build the frontend:
```bash
cd frontend
npm run build
```

2. Deploy the `dist/` folder to your hosting service

3. Set environment variables:
```env
VITE_API_URL=https://your-backend-url.com
```

### **Backend Deployment (Railway/Render/Heroku)**

1. Use the provided Dockerfile:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Set environment variables for production

3. Ensure database is properly configured

### **Docker Deployment**

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t ict-dashboard-backend ./backend
docker build -t ict-dashboard-frontend ./frontend
```

## 🧪 Testing

### **Backend Testing**
```bash
cd backend
python -m pytest tests/
```

### **Frontend Testing**
```bash
cd frontend
npm run test
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Backend won't start**
   - Check Python version (3.10+ required)
   - Ensure all dependencies are installed
   - Verify database initialization

2. **Frontend build errors**
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version (18+ required)

3. **Database errors**
   - Run database initialization: `python init_database.py`
   - Check database file permissions

4. **ML model errors**
   - Models will auto-generate on first use
   - Check if training data is available

### **Performance Optimization**

- **Database**: Regular cleanup of old entries
- **ML Models**: Retrain periodically with new data
- **Frontend**: Enable production build optimizations
- **Backend**: Use production ASGI server (Gunicorn + Uvicorn)

## 📈 Monitoring & Maintenance

### **System Health Checks**
- Monitor API response times
- Check database size and performance
- Review ML model accuracy metrics
- Monitor user activity and engagement

### **Regular Maintenance**
- Update dependencies regularly
- Backup database periodically
- Review and clean up old data
- Monitor system logs for errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push to your fork: `git push origin feature-name`
6. Create a pull request

## 📄 License

This project is developed as part of a research thesis on ICT impact assessment in medical college libraries. All rights reserved.

## 👥 Support & Contact

For technical support, bug reports, or feature requests:
- Create an issue in the repository
- Contact the development team
- Review the documentation and troubleshooting guide

## 🎯 Future Enhancements

- **Mobile App**: React Native mobile application
- **Advanced Analytics**: More sophisticated ML models
- **Real-time Notifications**: WebSocket-based updates
- **Multi-language Support**: Internationalization
- **Advanced Reporting**: More export formats and customization
- **Integration APIs**: Connect with external library systems

---

**Built with ❤️ for improving ICT infrastructure in medical college libraries**

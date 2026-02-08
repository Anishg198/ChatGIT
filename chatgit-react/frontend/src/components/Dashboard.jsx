import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const Dashboard = ({ metrics }) => {
    const [currentView, setCurrentView] = useState('files');
    const [topFiles, setTopFiles] = useState([]);
    const [topFunctions, setTopFunctions] = useState([]);
    const [topModules, setTopModules] = useState([]);

    useEffect(() => {
        const loadAnalysisData = async () => {
            try {
                const [filesRes, funcsRes, modsRes] = await Promise.all([
                    axios.get('${API_BASE_URL}/api/pagerank/files'),
                    axios.get('${API_BASE_URL}/api/pagerank/functions'),
                    axios.get('${API_BASE_URL}/api/pagerank/modules')
                ]);

                setTopFiles(filesRes.data);
                setTopFunctions(funcsRes.data);
                setTopModules(modsRes.data);
            } catch (err) {
                console.error("Failed to load dashboard metrics", err);
            }
        };

        if (metrics) {
            loadAnalysisData();
        }
    }, [metrics]);

    if (!metrics) return null;

    return (
        <div>
            <div className="header">
                <h1>CHATGIT</h1>
                <p className="subtitle">MULTI-LANGUAGE CODE ANALYSIS WITH AI</p>
                <p style={{ color: '#0000FF', fontSize: '12px', marginTop: '-10px' }}>
                    Supports Python • JavaScript • TypeScript • Java • Swift • C/C++
                </p>
            </div>

            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-label">TOTAL FILES</div>
                    <div className="stat-value">{metrics.total_files?.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">FUNCTIONS</div>
                    <div className="stat-value">{metrics.total_functions?.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">CLASSES</div>
                    <div className="stat-value">{metrics.total_classes?.toLocaleString()}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">PACKAGES</div>
                    <div className="stat-value">{metrics.total_packages?.toLocaleString()}</div>
                </div>
            </div>

            <h2>PAGERANK ANALYSIS</h2>

            <div className="tabs">
                <button
                    className={`tab ${currentView === 'files' ? 'active' : ''}`}
                    onClick={() => setCurrentView('files')}
                >
                    FILES
                </button>
                <button
                    className={`tab ${currentView === 'functions' ? 'active' : ''}`}
                    onClick={() => setCurrentView('functions')}
                >
                    FUNCTIONS
                </button>
                <button
                    className={`tab ${currentView === 'modules' ? 'active' : ''}`}
                    onClick={() => setCurrentView('modules')}
                >
                    MODULES
                </button>
            </div>

            <div className="tab-content">
                {currentView === 'files' && (
                    <div>
                        <h3>TOP FILES BY IMPORTANCE</h3>
                        {topFiles.map((item, index) => (
                            <div key={index} className="retro-list-item">
                                <span>{index + 1}. {item.name}</span>
                                <span>{item.score.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                )}

                {currentView === 'functions' && (
                    <div>
                        <h3>TOP FUNCTIONS BY IMPORTANCE</h3>
                        {topFunctions.map((func, index) => (
                            <div key={index} className="retro-list-item">
                                <span>{index + 1}. {func.name}</span>
                                <span>{func.score.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                )}

                {currentView === 'modules' && (
                    <div>
                        <h3>TOP MODULES BY IMPORTANCE</h3>
                        {topModules.map((mod, index) => (
                            <div key={index} className={`retro-list-item ${mod.is_local ? 'local' : ''}`}>
                                <span>{index + 1}. {mod.name} {mod.is_local ? '(local)' : ''}</span>
                                <span>{mod.score.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './config';
import './App.css';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Chat from './components/Chat';
import CallGraph from './components/CallGraph';
import StructureExplorer from './components/StructureExplorer';

function App() {
  const [activeRepo, setActiveRepo] = useState(null);
  const [loadingState, setLoadingState] = useState(false);
  const [repoMetrics, setRepoMetrics] = useState(null);
  const [conversationParams, setConversationParams] = useState([]);
  const [shouldEnhance, setShouldEnhance] = useState(true);
  const [displayGraph, setDisplayGraph] = useState(false);
  const [displayTree, setDisplayTree] = useState(false);

  useEffect(() => {
    // Check for existing session
    const restoreSession = async () => {
      try {
        const repoResponse = await axios.get(`${API_BASE_URL}/api/current_repo`);
        if (repoResponse.data.repo_name) {
          setActiveRepo(repoResponse.data.repo_name);
          const metricsResponse = await axios.get(`${API_BASE_URL}/api/stats`);
          setRepoMetrics(metricsResponse.data);
          const historyResponse = await axios.get(`${API_BASE_URL}/api/chat/history`);
          setConversationParams(historyResponse.data);
        }
      } catch (err) {
        console.error("Failed to restore session", err);
      }
    };
    restoreSession();
  }, []);

  const initiateRepoLoad = async (repoUrl) => {
    setLoadingState(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/load_repo`, { github_url: repoUrl });
      if (response.data.status === 'success') {
        setActiveRepo(response.data.repo_name);
        const metrics = await axios.get(`${API_BASE_URL}/api/stats`);
        setRepoMetrics(metrics.data);
        setConversationParams([]);
      }
    } catch (err) {
      alert("Failed to load repo: " + err.message);
    } finally {
      setLoadingState(false);
    }
  };

  const resetRepository = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/clear_repo`);
      setActiveRepo(null);
      setRepoMetrics(null);
      setConversationParams([]);
    } catch (err) {
      console.error("Failed to clear repo", err);
    }
  };

  return (
    <div className="app-container">
      {loadingState && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <h2>Processing Repository...</h2>
          <p>Cloning, analyzing structure, and indexing code. Please wait.</p>
        </div>
      )}
      <Sidebar
        onRepoLoad={initiateRepoLoad}
        onRepoClear={resetRepository}
        activeRepoName={activeRepo}
        isProcessing={loadingState}
        enhanceEnabled={shouldEnhance}
        toggleEnhance={setShouldEnhance}
        graphEnabled={displayGraph}
        toggleGraph={setDisplayGraph}
        treeEnabled={displayTree}
        toggleTree={setDisplayTree}
      />
      <div className="main-content">
        {activeRepo ? (
          <>
            <Dashboard metrics={repoMetrics} />
            {displayTree && <StructureExplorer />}
            {displayGraph && <CallGraph />}
            <h2 style={{ marginTop: '40px', marginBottom: '20px' }}>CHAT</h2>
            <Chat chatLog={conversationParams} codeEnhancement={shouldEnhance} />
          </>
        ) : (
          <div style={{ display: 'flex', height: '100%', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', color: '#0000FF' }}>
            <h1>WELCOME TO CHATGIT</h1>
            <p>Select a repository from the sidebar to start.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

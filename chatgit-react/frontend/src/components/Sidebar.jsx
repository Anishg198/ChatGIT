import React, { useState } from 'react';

const Toggle = ({ checked, onChange }) => (
  <label className="toggle">
    <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
    <span className="toggle-track" />
  </label>
);

const Sidebar = ({
  onRepoLoad,
  onRepoClear,
  activeRepoName,
  isProcessing,
  enhanceEnabled,
  toggleEnhance,
  graphEnabled,
  toggleGraph,
  treeEnabled,
  toggleTree,
  hitsEnabled,
  toggleHits,
}) => {
  const [repoInput, setRepoInput] = useState('');

  const handleLoad = () => { if (repoInput.trim()) onRepoLoad(repoInput.trim()); };
  const handleKey  = (e) => { if (e.key === 'Enter') handleLoad(); };

  return (
    <div className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">⬡</div>
        <h1>ChatGIT</h1>
      </div>

      {/* Repo input */}
      <div className="sidebar-section">
        <span className="sidebar-label">GitHub Repository</span>
        <div className="sidebar-input-wrap">
          <span className="sidebar-input-icon">⌂</span>
          <input
            type="text"
            placeholder="https://github.com/owner/repo"
            value={repoInput}
            onChange={e => setRepoInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={isProcessing}
          />
        </div>
        <button
          className="btn-primary"
          onClick={handleLoad}
          disabled={isProcessing || !repoInput.trim()}
        >
          {isProcessing
            ? <><span className="loading-ring" style={{width:14,height:14,borderWidth:2}} /> Analyzing…</>
            : <>⬆ Load Repository</>}
        </button>
      </div>

      {/* Active repo */}
      {activeRepoName && (
        <div className="sidebar-section">
          <span className="sidebar-label">Active Repository</span>
          <div className="repo-badge">
            <div className="repo-badge-name">
              <span className="repo-badge-dot" />
              {activeRepoName}
            </div>
          </div>
          <button className="btn-ghost" onClick={onRepoClear}>
            ✕ Clear Repository
          </button>
        </div>
      )}

      {/* View toggles */}
      <div className="sidebar-section">
        <span className="sidebar-label">Views</span>
        <div className="toggle-row">
          <span className="toggle-label"><span className="icon">⏱</span> File Tree (AST)</span>
          <Toggle checked={treeEnabled} onChange={toggleTree} />
        </div>
        <div className="toggle-row">
          <span className="toggle-label"><span className="icon">⬡</span> Call Graph</span>
          <Toggle checked={graphEnabled} onChange={toggleGraph} />
        </div>
        <div className="toggle-row">
          <span className="toggle-label"><span className="icon">◎</span> HITS Analysis</span>
          <Toggle checked={hitsEnabled} onChange={toggleHits} />
        </div>
      </div>

      {/* Feature toggles */}
      <div className="sidebar-section">
        <span className="sidebar-label">Features</span>
        <div className="toggle-row">
          <span className="toggle-label"><span className="icon">✦</span> Snippet Enhancement</span>
          <Toggle checked={enhanceEnabled} onChange={toggleEnhance} />
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 'auto', paddingTop: 16, borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
          <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>ChatGIT v2.0</div>
          Multi-turn conversational repo intelligence with session memory, intent routing &amp; call-graph augmentation.
        </div>
      </div>
    </div>
  );
};

export default Sidebar;

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

/* Animated counter hook */
function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!target) return;
    const start = Date.now();
    const tick = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(ease * target));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, duration]);
  return value;
}

const StatCard = ({ icon, label, value, delay, color }) => {
  const animated = useCountUp(value);
  return (
    <div className="stat-card" style={{ animationDelay: `${delay}s` }}>
      <span className="stat-icon">{icon}</span>
      <div className="stat-value" style={{ color: color || 'var(--text-primary)' }}>
        {animated.toLocaleString()}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
};

const RankItem = ({ rank, name, score, maxScore, isLocal, delay, colorClass }) => {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
  return (
    <div className="rank-item" style={{ animationDelay: `${delay}s` }}>
      <span className="rank-num">{rank}</span>
      <span className={`rank-name ${isLocal ? 'local' : ''}`} title={name}>
        {name.split('/').pop() || name}
      </span>
      <div className="rank-bar-wrap">
        <div className="rank-bar">
          <div className="rank-bar-fill" style={{ width: `${pct}%` }} />
        </div>
        <span className="rank-score">
          {typeof score === 'number' && score < 0.001
            ? score.toExponential(1)
            : typeof score === 'number'
              ? score.toFixed(4)
              : score}
        </span>
      </div>
    </div>
  );
};

const Dashboard = ({ metrics, showHits }) => {
  const [view, setView] = useState('files');
  const [topFiles, setTopFiles]       = useState([]);
  const [topFunctions, setTopFunctions] = useState([]);
  const [topModules, setTopModules]   = useState([]);
  const [hitsData, setHitsData]       = useState(null);

  useEffect(() => {
    if (!metrics) return;
    const load = async () => {
      try {
        const [fR, fnR, mR] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/pagerank/files`),
          axios.get(`${API_BASE_URL}/api/pagerank/functions`),
          axios.get(`${API_BASE_URL}/api/pagerank/modules`),
        ]);
        setTopFiles(fR.data || []);
        setTopFunctions(fnR.data || []);
        setTopModules(mR.data || []);
      } catch { /* silent */ }
    };
    load();
  }, [metrics]);

  useEffect(() => {
    if (!metrics || !showHits) return;
    axios.get(`${API_BASE_URL}/api/hits`)
      .then(r => setHitsData(r.data))
      .catch(() => {});
  }, [metrics, showHits]);

  if (!metrics) return null;

  const tabs = [
    { id: 'files',     label: '📄 Files' },
    { id: 'functions', label: '⚙ Functions' },
    { id: 'modules',   label: '📦 Modules' },
    ...(showHits ? [{ id: 'hits', label: '◎ HITS' }] : []),
  ];

  const maxScore = (list) =>
    list.length ? Math.max(...list.map(x => x.score ?? 0)) : 1;

  return (
    <div>
      {/* Header */}
      <div className="dash-header">
        <div className="dash-header-top">
          <div>
            <div className="dash-title">Repository Intelligence</div>
            <div className="dash-subtitle">AI-Powered Code Analysis</div>
          </div>
        </div>
        <div className="lang-chips">
          {['py','js','ts','java','go','cpp'].map(l => (
            <span key={l} className={`lang-chip ${l}`}>{l.toUpperCase()}</span>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <StatCard icon="🗂" label="Total Files"     value={metrics.total_files}     delay={0.05} color="var(--accent-light)" />
        <StatCard icon="⚙" label="Functions"       value={metrics.total_functions}  delay={0.10} color="var(--purple)" />
        <StatCard icon="◻" label="Classes"         value={metrics.total_classes}    delay={0.15} color="var(--cyan)" />
        <StatCard icon="📦" label="Packages"        value={metrics.total_packages}   delay={0.20} color="var(--amber)" />
      </div>

      {/* PageRank / HITS */}
      <div className="section-header">
        <h2>Graph Analysis</h2>
        <span className="section-badge">NetworkX</span>
      </div>

      <div className="tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab ${view === t.id ? 'active' : ''}`}
            onClick={() => setView(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {view === 'files' && (
          <div className="rank-list">
            {topFiles.map((item, i) => (
              <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                maxScore={maxScore(topFiles)} delay={i * 0.04} />
            ))}
          </div>
        )}

        {view === 'functions' && (
          <div className="rank-list">
            {topFunctions.map((item, i) => (
              <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                maxScore={maxScore(topFunctions)} delay={i * 0.04} />
            ))}
          </div>
        )}

        {view === 'modules' && (
          <div className="rank-list">
            {topModules.map((item, i) => (
              <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                isLocal={item.is_local} maxScore={maxScore(topModules)} delay={i * 0.04} />
            ))}
          </div>
        )}

        {view === 'hits' && hitsData && (
          <div>
            {/* File HITS */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ marginBottom: 12, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>
                File-Level HITS
              </div>
              <div className="hits-col">
                <div>
                  <div className="hits-section-title">
                    <span className="hits-hub-dot" /> Hubs
                    <span style={{ fontWeight: 400, fontSize: 10, color: 'var(--text-muted)' }}>
                      &nbsp;— entry points / orchestrators
                    </span>
                  </div>
                  <div className="rank-list">
                    {(hitsData.files?.hubs || []).map((item, i) => (
                      <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                        maxScore={Math.max(...(hitsData.files?.hubs||[]).map(x=>x.score),0.001)}
                        delay={i*0.04} />
                    ))}
                    {!(hitsData.files?.hubs?.length) && (
                      <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No hub data</div>
                    )}
                  </div>
                </div>
                <div>
                  <div className="hits-section-title">
                    <span className="hits-auth-dot" /> Authorities
                    <span style={{ fontWeight: 400, fontSize: 10, color: 'var(--text-muted)' }}>
                      &nbsp;— core utilities / widely imported
                    </span>
                  </div>
                  <div className="rank-list">
                    {(hitsData.files?.authorities || []).map((item, i) => (
                      <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                        maxScore={Math.max(...(hitsData.files?.authorities||[]).map(x=>x.score),0.001)}
                        delay={i*0.04} />
                    ))}
                    {!(hitsData.files?.authorities?.length) && (
                      <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No authority data</div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Function HITS */}
            <div>
              <div style={{ marginBottom: 12, fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>
                Function-Level HITS
              </div>
              <div className="hits-col">
                <div>
                  <div className="hits-section-title">
                    <span className="hits-hub-dot" /> Hubs
                    <span style={{ fontWeight: 400, fontSize: 10, color: 'var(--text-muted)' }}>
                      &nbsp;— call many functions
                    </span>
                  </div>
                  <div className="rank-list">
                    {(hitsData.functions?.hubs || []).map((item, i) => (
                      <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                        maxScore={Math.max(...(hitsData.functions?.hubs||[]).map(x=>x.score),0.001)}
                        delay={i*0.04} />
                    ))}
                    {!(hitsData.functions?.hubs?.length) && (
                      <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No hub data</div>
                    )}
                  </div>
                </div>
                <div>
                  <div className="hits-section-title">
                    <span className="hits-auth-dot" /> Authorities
                    <span style={{ fontWeight: 400, fontSize: 10, color: 'var(--text-muted)' }}>
                      &nbsp;— called by many hubs
                    </span>
                  </div>
                  <div className="rank-list">
                    {(hitsData.functions?.authorities || []).map((item, i) => (
                      <RankItem key={i} rank={i+1} name={item.name} score={item.score}
                        maxScore={Math.max(...(hitsData.functions?.authorities||[]).map(x=>x.score),0.001)}
                        delay={i*0.04} />
                    ))}
                    {!(hitsData.functions?.authorities?.length) && (
                      <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No authority data</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {view === 'hits' && !hitsData && (
          <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '20px 0' }}>
            Loading HITS analysis…
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

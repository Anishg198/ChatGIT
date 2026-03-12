import React, { useState } from 'react';

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
    toggleTree
}) => {
    const [repoInput, setRepoInput] = useState('');

    const triggerLoad = () => {
        if (repoInput) onRepoLoad(repoInput);
    };

    return (
        <div className="sidebar">
            <h1>CHATGIT</h1>
            <div className="sidebar-section">
                <h3>GITHUB REPOSITORY</h3>
                <input
                    type="text"
                    placeholder="https://github.com/owner/repo"
                    value={repoInput}
                    onChange={(e) => setRepoInput(e.target.value)}
                    disabled={isProcessing}
                />
                <button onClick={triggerLoad} disabled={isProcessing || !repoInput}>
                    {isProcessing ? 'PROCESSING...' : 'LOAD REPOSITORY'}
                </button>
            </div>

            <div className="sidebar-section">
                <h3>OPTIONS</h3>
                <label className="sidebar-checkbox">
                    <input
                        type="checkbox"
                        checked={treeEnabled}
                        onChange={(e) => toggleTree(e.target.checked)}
                    />
                    SHOW FILE TREE (AST)
                </label>
                <label className="sidebar-checkbox">
                    <input
                        type="checkbox"
                        checked={graphEnabled}
                        onChange={(e) => toggleGraph(e.target.checked)}
                    />
                    SHOW CALL GRAPH
                </label>
                <label className="sidebar-checkbox">
                    <input
                        type="checkbox"
                        checked={enhanceEnabled}
                        onChange={(e) => toggleEnhance(e.target.checked)}
                    />
                    ENHANCE SNIPPETS
                </label>
            </div>

            {activeRepoName && (
                <div className="sidebar-section">
                    <h3>CURRENT REPOSITORY</h3>
                    <div className="current-repo">
                        <code>{activeRepoName}</code>
                    </div>
                    <button onClick={onRepoClear} style={{ marginTop: '10px' }}>
                        CLEAR REPOSITORY
                    </button>
                </div>
            )}
        </div>
    );
};

export default Sidebar;

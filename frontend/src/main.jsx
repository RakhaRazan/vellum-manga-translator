import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

class RootErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error: error || new Error('Unknown application error') };
  }

  render() {
    if (this.state.error) {
      const msg = this.state.error?.message || String(this.state.error);
      const stack = this.state.error?.stack || '';
      return (
        <div
          style={{
            minHeight: '100vh',
            background: '#161618',
            color: '#ececf1',
            padding: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              maxWidth: '720px',
              width: '100%',
              background: '#1f1f23',
              border: '1px solid #ef4444',
              borderRadius: '8px',
              padding: '20px',
            }}
          >
            <h2 style={{ margin: '0 0 8px', fontSize: '15px', color: '#f87171' }}>
              Vellum UI Runtime Error
            </h2>
            <p style={{ margin: '0 0 12px', fontSize: '13px', color: '#e4e4e7' }}>
              {msg}
            </p>
            {stack && (
              <pre
                style={{
                  margin: '0 0 14px',
                  padding: '10px',
                  background: '#111113',
                  borderRadius: '6px',
                  fontSize: '11px',
                  overflowX: 'auto',
                  color: '#a1a1aa',
                  maxHeight: '260px',
                }}
              >
                {stack}
              </pre>
            )}
            <button
              type="button"
              onClick={() => window.location.reload()}
              style={{
                background: '#f59e0b',
                color: '#111113',
                border: 'none',
                borderRadius: '6px',
                padding: '7px 14px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Reload Vellum
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>
);

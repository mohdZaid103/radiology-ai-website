import { useEffect, useState } from 'react'
import { Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom'
import { Activity, Brain, ClipboardList, LayoutDashboard, LogOut, ShieldCheck, UploadCloud, FileImage, Menu } from 'lucide-react'
import api from './api'

function Auth({ mode, onAuth }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const nav = useNavigate()
  async function submit(e) {
    e.preventDefault(); setError(''); setBusy(true)
    try {
      const { data } = await api.post(`/auth/${mode}`, { email, password })
      localStorage.setItem('radiology_token', data.token); onAuth(data.user); nav('/')
    } catch (err) { setError(err.response?.data?.detail || err.response?.data?.message || 'Unable to authenticate') }
    finally { setBusy(false) }
  }
  return <div className="auth-page"><div className="auth-card">
    <div className="brand-mark"><Activity /></div><p className="eyebrow">RADIOLOGY AI ASSISTANT</p>
    <h1>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
    <p className="muted">Research-focused chest X-ray analysis workspace.</p>
    <form onSubmit={submit} className="form">
      <label>Email address<input type="email" required value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com" /></label>
      <label>Password<input type="password" minLength="8" required value={password} onChange={e=>setPassword(e.target.value)} placeholder="At least 8 characters" /></label>
      {error && <div className="error">{error}</div>}
      <button className="primary full" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
    </form>
    <p className="switch-auth">{mode === 'login' ? "New here?" : 'Already registered?'} <Link to={mode === 'login' ? '/register' : '/login'}>{mode === 'login' ? 'Create an account' : 'Sign in'}</Link></p>
  </div></div>
}

function Shell({ user, onLogout, children }) {
  const [open, setOpen] = useState(false)
  const nav = [{to:'/',label:'Overview',icon:LayoutDashboard},{to:'/analyze',label:'X-ray analysis',icon:UploadCloud},{to:'/history',label:'Scan history',icon:ClipboardList}]
  return <div className="layout">
    <aside className={open?'sidebar open':'sidebar'}>
      <div className="brand"><div className="brand-mark small"><Activity/></div><div><b>Radiology AI</b><small>ASSISTANT</small></div></div>
      <div className="nav-label">WORKSPACE</div>
      {nav.map(n=><Link onClick={()=>setOpen(false)} className="nav-link" to={n.to} key={n.to}><n.icon size={18}/>{n.label}</Link>)}
      <div className="sidebar-bottom"><div className="user-chip"><div className="avatar">{user?.email?.[0]?.toUpperCase()}</div><div className="user-meta"><b>{user?.email}</b><small>Research account</small></div></div><button className="logout" onClick={onLogout}><LogOut size={16}/> Sign out</button></div>
    </aside>
    <main className="main"><header className="topbar"><button className="mobile-menu" onClick={()=>setOpen(!open)}><Menu/></button><span>Research workspace</span><div className="top-right"><span className="status-dot"/> System interface</div></header>{children}</main>
  </div>
}

function Dashboard({ user, health }) {
  return <div className="content">
    <div className="page-heading"><div><p className="eyebrow">WORKSPACE OVERVIEW</p><h1>Good to see you</h1><p className="muted">Your chest X-ray research workspace at a glance.</p></div><Link to="/analyze" className="primary link-button"><UploadCloud size={17}/> New analysis</Link></div>
    <div className="welcome-banner"><div><span className="banner-tag"><ShieldCheck size={14}/> RESEARCH ENVIRONMENT</span><h2>Explainable chest X-ray classification</h2><p>Explore model predictions across four dataset-defined classes, with visual explanations from Grad-CAM.</p><Link to="/analyze" className="banner-link">Start an analysis <span>→</span></Link></div><div className="banner-art"><div className="scan-ring"><Activity size={56}/></div></div></div>
    <div className="section-title"><h2>System status</h2><span className="muted">Live API status</span></div>
    <div className="stats-grid">
      <div className="stat-card"><div className="stat-icon blue"><Brain/></div><span>Available model branches</span><strong>{health?.available_branches?.length ?? '—'}</strong><small>{health?.available_branches?.join(' + ') || 'No trained weights detected'}</small></div>
      <div className="stat-card"><div className="stat-icon teal"><Activity/></div><span>Model preset</span><strong className="stat-text">{health?.model_preset || '—'}</strong><small>Configured preprocessing</small></div>
      <div className="stat-card"><div className="stat-icon violet"><ShieldCheck/></div><span>Database</span><strong className="stat-text">{health?.database_connected ? 'Connected' : 'Unavailable'}</strong><small>Account and scan history</small></div>
    </div>
    <div className="section-title"><h2>Supported classification labels</h2><span className="muted">4 classes</span></div>
    <div className="class-grid">{['Normal','COVID-19','Pneumonia','Tuberculosis'].map((c,i)=><div className="class-card" key={c}><div className={`class-number c${i}`}>0{i+1}</div><b>{c}</b><small>Dataset-defined label</small></div>)}</div>
    <div className="disclaimer"><ShieldCheck size={18}/><div><b>Research prototype — not for clinical use</b><p>Outputs are not diagnoses, triage decisions or treatment recommendations. A qualified clinician must review any image.</p></div></div>
  </div>
}

function Analyze() {
  const [file,setFile] = useState(null), [branch,setBranch] = useState('cnn'), [health,setHealth] = useState(null)
  const [result,setResult] = useState(null), [error,setError] = useState(''), [busy,setBusy] = useState(false)
  useEffect(()=>{api.get('/health').then(r=>setHealth(r.data)).catch(()=>{})},[])
  async function submit(e) {
    e.preventDefault(); if(!file)return
    setBusy(true);setError('');setResult(null)
    const form=new FormData();form.append('file',file);form.append('branch',branch)
    try { const {data}=await api.post('/predict',form,{headers:{'Content-Type':'multipart/form-data'}});setResult(data) }
    catch(err){setError(err.response?.data?.detail || 'Prediction failed. Check that the backend and trained model are available.')}
    finally{setBusy(false)}
  }
  return <div className="content">
    <div className="page-heading"><div><p className="eyebrow">AI WORKBENCH</p><h1>X-ray analysis</h1><p className="muted">Upload a chest X-ray image to run the selected research model.</p></div></div>
    <div className="analysis-grid"><section className="panel upload-panel"><div className="panel-head"><div><h2>Upload image</h2><p className="muted">PNG, JPG, JPEG, BMP or TIFF · Max 10 MB</p></div><FileImage/></div>
      <form onSubmit={submit}>
        <label className="dropzone"><input type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={e=>{setFile(e.target.files[0]);setResult(null)}}/>
          <div className="upload-icon"><UploadCloud/></div><b>{file ? file.name : 'Choose a chest X-ray image'}</b><span>{file ? `${(file.size/1024/1024).toFixed(2)} MB selected` : 'Browse files or drop an image here'}</span>
        </label>
        {file && <div className="preview"><img src={URL.createObjectURL(file)} alt="Selected X-ray preview"/><button type="button" className="text-button" onClick={()=>setFile(null)}>Remove image</button></div>}
        <label className="field-label">Model branch<select value={branch} onChange={e=>setBranch(e.target.value)}><option value="cnn">DenseNet121 CNN (Grad-CAM)</option><option value="svc">Deep features + SVC</option></select></label>
        {branch==='svc' && <p className="hint">SVC does not produce Grad-CAM. Its decision scores are not probabilities unless calibrated during training.</p>}
        {error && <div className="error">{error}</div>}
        <button disabled={!file||busy} className="primary full">{busy?'Analyzing image…':'Run analysis'}</button>
      </form>
      {!health?.available_branches?.length && <div className="hint warning">No trained model weights are currently available. Train the CNN model first, then restart the backend.</div>}
    </section>
    <section className="panel result-panel"><div className="panel-head"><div><h2>Analysis result</h2><p className="muted">Prediction and explainability output</p></div><Brain/></div>
      {!result ? <div className="empty-result"><div className="empty-icon"><Activity/></div><b>Waiting for an image</b><p>Your model output will appear here after analysis.</p></div> :
      <div className="result-body"><span className="result-label">PREDICTED CLASS</span><h2>{result.predicted_class}</h2><div className="result-meta"><span>{result.branch.toUpperCase()} branch</span><span>{result.latency_ms} ms</span></div>
        {result.class_scores && <div className="scores">{Object.entries(result.class_scores).map(([name,val])=><div className="score-row" key={name}><div><span>{name}</span><b>{(val*100).toFixed(1)}%</b></div><div className="bar"><i style={{width:`${val*100}%`}}/></div></div>)}</div>}
        {result.decision_scores && <div className="hint">SVC decision margins: {Object.entries(result.decision_scores).map(([k,v])=>`${k}: ${Number(v).toFixed(3)}`).join(' · ')}</div>}
        {result.gradcam_overlay && <div className="gradcam"><h3>Grad-CAM overlay</h3><img src={result.gradcam_overlay} alt="Grad-CAM visualization"/><p>{result.gradcam_note}</p></div>}
        <div className="hint">{result.score_note}</div><small className="muted">Scan ID: {result.scan_id}</small>
      </div>}
    </section></div>
    <div className="disclaimer"><ShieldCheck size={18}/><div><b>Research prototype — not for clinical use</b><p>Model scores are not clinical probabilities. This result must not be used for diagnosis, triage or treatment.</p></div></div>
  </div>
}

function History() {
  const [items,setItems]=useState([]),[error,setError]=useState('')
  useEffect(()=>{api.get('/history').then(r=>setItems(r.data)).catch(e=>setError(e.response?.data?.detail||'Could not load scan history'))},[])
  return <div className="content"><div className="page-heading"><div><p className="eyebrow">YOUR ACTIVITY</p><h1>Scan history</h1><p className="muted">Previously completed analyses associated with your account.</p></div><Link className="primary link-button" to="/analyze">New analysis</Link></div>
    <div className="panel history-panel"><div className="panel-head"><div><h2>Recent scans</h2><p className="muted">Most recent 30 records</p></div><ClipboardList/></div>
      {error&&<div className="error">{error}</div>}
      {!items.length&&!error?<div className="empty-result"><ClipboardList/><b>No scans yet</b><p>Your completed analyses will be listed here.</p></div>:
      <div className="table-wrap"><table><thead><tr><th>File</th><th>Prediction</th><th>Model</th><th>Processing time</th><th>Date</th></tr></thead><tbody>{items.map(x=><tr key={x.id}><td>{x.filename}</td><td><span className="pill">{x.predicted_class}</span></td><td>{x.branch.toUpperCase()}</td><td>{x.latency_ms??'—'} ms</td><td>{new Date(x.created_at).toLocaleString()}</td></tr>)}</tbody></table></div>}
    </div></div>
}

export default function App() {
  const [user,setUser]=useState(null),[checking,setChecking]=useState(true),[health,setHealth]=useState(null)
  useEffect(()=>{
    const token=localStorage.getItem('radiology_token')
    if(!token){setChecking(false);return}
    Promise.all([api.get('/auth/me'),api.get('/health')]).then(([u,h])=>{setUser(u.data);setHealth(h.data)}).catch(()=>localStorage.removeItem('radiology_token')).finally(()=>setChecking(false))
  },[])
  useEffect(()=>{if(user)api.get('/health').then(r=>setHealth(r.data)).catch(()=>{})},[user])
  const logout=()=>{localStorage.removeItem('radiology_token');setUser(null)}
  if(checking)return <div className="loading">Loading workspace…</div>
  return <Routes>
    <Route path="/login" element={user?<Navigate to="/"/>:<Auth mode="login" onAuth={setUser}/>} />
    <Route path="/register" element={user?<Navigate to="/"/>:<Auth mode="register" onAuth={setUser}/>} />
    <Route path="/*" element={user?<Shell user={user} onLogout={logout}><Routes><Route path="/" element={<Dashboard user={user} health={health}/>} /><Route path="/analyze" element={<Analyze/>}/><Route path="/history" element={<History/>}/><Route path="*" element={<Navigate to="/"/>}/></Routes></Shell>:<Navigate to="/login"/>}/>
  </Routes>
}

import { FormEvent, useEffect, useState } from 'react'

import { me } from '../api/auth'
import type { CurrentUser } from '../api/types'
import { JsonBlock } from '../components/common/JsonBlock'
import { PageHeader } from '../components/common/PageHeader'

export function SettingsPage() {
  const [apiBaseUrl, setApiBaseUrl] = useState(localStorage.getItem('apiBaseUrl') || 'http://127.0.0.1:8000/api/v1')
  const [user, setUser] = useState<CurrentUser | null>(null)
  useEffect(() => { me().then(setUser).catch(() => setUser(null)) }, [])
  function save(event: FormEvent) { event.preventDefault(); localStorage.setItem('apiBaseUrl', apiBaseUrl) }
  return (
    <section className="workbench-page">
      <PageHeader title="设置" description="运行配置和治理规则。" />
      <div className="split">
        <div className="stack">
          <form className="panel stack" onSubmit={save}><label>API Base URL<input className="input" value={apiBaseUrl} onChange={(event) => setApiBaseUrl(event.target.value)} /></label><button className="button" style={{ width: 'fit-content' }}>保存设置</button></form>
          <div className="panel"><h2>Feed 评分公式</h2><pre className="code-block">final_score ={"\n"}0.30 * personal_relevance{"\n"}+ 0.20 * novelty{"\n"}+ 0.15 * cross_domain_distance{"\n"}+ 0.15 * opportunity_value{"\n"}+ 0.10 * source_credibility{"\n"}+ 0.10 * actionability</pre></div>
          <div className="panel"><h2>权限等级</h2><pre className="code-block">L0_READ_ONLY：只读，可自动执行{"\n"}L1_DRAFT：生成草稿{"\n"}L2_LOCAL_WRITE：本地写入{"\n"}L3_EXTERNAL_WRITE：必须审批{"\n"}L4_HIGH_RISK：默认阻断</pre></div>
        </div>
        <aside className="panel"><h2>当前用户</h2><p>{user?.email || '未读取到用户信息'}</p><p className="muted">当前阶段不执行真实邮件发送、浏览器控制、视频评论、表单提交、Exa 或 Neo4j。</p><details><summary>技术详情</summary><JsonBlock value={user} /></details></aside>
      </div>
    </section>
  )
}

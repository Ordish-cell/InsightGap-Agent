import { NavLink } from 'react-router-dom'

const links: Array<[string, string]> = [
  ['Feed', '/feed'],
  ['Research', '/research'],
  ['Agent', '/agent'],
  ['Artifacts', '/artifacts'],
  ['Memory', '/memory'],
  ['Skills', '/skills'],
  ['Approvals', '/approvals'],
  ['MCP Tools', '/mcp'],
  ['Settings', '/settings'],
]

export function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">IG</span>
        <div>
          <strong>Agent OS</strong>
          <small>Information gap workbench</small>
        </div>
      </div>
      <nav>
        {links.map(([label, href]) => (
          <NavLink key={href} to={href} className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}

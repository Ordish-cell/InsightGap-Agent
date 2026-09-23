import type { CSSProperties } from 'react'

const paths: Record<string, string> = {
  chat: 'M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9H13a8.5 8.5 0 0 1 8 8v.5Z',
  feed: 'M4 5h16M4 12h10M4 19h7M18 10v10m-4-4 4 4 4-4',
  research: 'm21 21-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0ZM6 10h8M10 6v8',
  artifact: 'M5 3h9l5 5v13H5V3Zm9 0v6h5M8 13h8M8 17h6',
  memory:
    'M9 3a3 3 0 0 0-3 3 4 4 0 0 0-2 7 4 4 0 0 0 2 7 3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Zm6 0a3 3 0 0 1 3 3 4 4 0 0 1 2 7 4 4 0 0 1-2 7 3 3 0 0 1-6 0M6 9h3M15 15h4',
  skill: 'm12 3 2.6 5.4 6 .9-4.3 4.2 1 6L12 16.7 6.7 19.5l1-6L3.4 9.3l6-.9L12 3Z',
  approval: 'm12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6',
  audit: 'M4 4h16v16H4V4Zm4 4h8M8 12h5M8 16h8',
  code: 'm8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18',
  settings:
    'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm-2-6h4l1 3 3 1 3 1v4l-3 2-1 3-1 4h-4l-2-3-3-1-4-1v-4l3-2 1-3 1-4Z',
  user: 'M20 21v-2a6 6 0 0 0-6-6h-4a6 6 0 0 0-6 6v2M16 6a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
  plus: 'M12 5v14M5 12h14',
  menu: 'M4 6h16M4 12h16M4 18h16',
  close: 'm6 6 12 12M6 18 18 6',
  chevron: 'm9 5 7 7-7 7',
  arrow: 'M5 12h14m-6-6 6 6-6 6',
  collapse: 'M4 4h16v16H4V4Zm5 0v16m7-13-3 3 3 3',
  refresh: 'M20 7v5h-5M4 17v-5h5M6 6a8 8 0 0 1 13 2M5 16a8 8 0 0 0 13 2',
  logout: 'M9 3H4v18h5m5-14 5 5-5 5m-6-5h12',
  check: 'm5 12 4 4L19 6',
  spark: 'm12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z',
}
export function Icon({
  name,
  size = 20,
  style,
}: {
  name: keyof typeof paths
  size?: number
  style?: CSSProperties
}) {
  return (
    <svg
      className="ui-icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={style}
    >
      <path d={paths[name] || paths.chat} />
    </svg>
  )
}

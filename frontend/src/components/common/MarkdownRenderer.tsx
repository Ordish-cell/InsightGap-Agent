import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'

type MarkdownRendererProps = {
  content: string
}

function normalizeMarkdown(input: string): string {
  return (input || '')
    .replace(/\r\n/g, '\n')
    // Strip whitespace-only lines
    .replace(/[ \t]+\n/g, '\n')
    // Split multiple ✅ / ☑️ / ☐ / □ options crammed onto one line
    .replace(/\s+(?=(✅|☑️|☐|□)\s*)/g, '\n')
    // Insert blank line after a colon that directly precedes a checklist line
    .replace(/([：:])\s*\n?(?=(✅|☑️|☐|□)\s*)/g, '$1\n\n')
    // Collapse 3+ consecutive newlines to 2
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const normalized = normalizeMarkdown(content)
  if (!normalized) return null

  return (
    <div className="md-chat">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          h1: ({ children }) => <h1 className="md-h1">{children}</h1>,
          h2: ({ children }) => <h2 className="md-h2">{children}</h2>,
          h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
          h4: ({ children }) => <h4 className="md-h4">{children}</h4>,
          p: ({ children }) => <p className="md-p">{children}</p>,
          ul: ({ children }) => <ul className="md-ul">{children}</ul>,
          ol: ({ children }) => <ol className="md-ol">{children}</ol>,
          li: ({ children }) => <li className="md-li">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="md-blockquote">{children}</blockquote>
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer" className="md-a">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="md-table-wrap">
              <table className="md-table">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="md-th">{children}</th>,
          td: ({ children }) => <td className="md-td">{children}</td>,
          pre: ({ children }) => <pre className="md-pre">{children}</pre>,
          code: ({ className, children, ...props }) => {
            const isBlock =
              typeof className === 'string' && className.startsWith('language-')
            if (isBlock) {
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              )
            }
            return (
              <code className="md-code" {...props}>
                {children}
              </code>
            )
          },
          br: () => <br className="md-br" />,
          hr: () => null,
          strong: ({ children }) => (
            <strong className="md-strong">{children}</strong>
          ),
          em: ({ children }) => <em className="md-em">{children}</em>,
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  )
}

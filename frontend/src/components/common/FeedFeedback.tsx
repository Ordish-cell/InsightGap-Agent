import { useEffect, useRef, useState } from 'react'
import { feedback } from '../../api/feed'
import { ActionNotice } from './ActionNotice'
import { usePresence } from './motion'
import { useAction } from './useAction'

export function FeedFeedback({ cardId }: { cardId: number }) {
  const [open, setOpen] = useState(false)
  const present = usePresence(open)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const action = useAction()
  useEffect(() => {
    if (!open) return
    root.current?.querySelector<HTMLElement>('.feedback-menu button')?.focus()
    const close = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false)
        trigger.current?.focus()
      }
    }
    window.addEventListener('mousedown', close)
    window.addEventListener('keydown', key)
    return () => {
      window.removeEventListener('mousedown', close)
      window.removeEventListener('keydown', key)
    }
  }, [open])
  return (
    <>
      <div
        className="feedback-control"
        ref={root}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false)
        }}
      >
        <button
          ref={trigger}
          className="button ghost"
          aria-expanded={open}
          disabled={action.busy}
          onClick={() => setOpen((value) => !value)}
        >
          {action.busy ? '提交中…' : '反馈'}
        </button>
        {present && (
          <div className={`feedback-menu ${open ? '' : 'is-closing'}`} inert={!open}>
            {Object.entries({
              save: '保存',
              useful: '有用',
              ignore: '忽略',
              not_relevant: '不相关',
            }).map(([value, label]) => (
              <button
                key={value}
                disabled={action.busy}
                onClick={() => {
                  setOpen(false)
                  trigger.current?.focus()
                  void action.run(() => feedback(cardId, { action: value }), `已反馈：${label}。`)
                }}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
      {action.message && (
        <div className="feedback-result">
          <ActionNotice message={action.message} error={action.failed} />
        </div>
      )}
    </>
  )
}

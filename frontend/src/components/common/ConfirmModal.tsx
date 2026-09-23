import { createPortal } from 'react-dom'
import { useEffect, useId, useRef } from 'react'

import { usePresence } from './motion'

type ConfirmModalProps = {
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmModal({
  open,
  title,
  message,
  confirmLabel = '确认',
  cancelLabel = '取消',
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmModalProps) {
  const confirmRef = useRef<HTMLButtonElement | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const cancel = useRef(onCancel)
  cancel.current = onCancel
  const titleId = useId()
  const present = usePresence(open)
  const lastContent = useRef({ title, message, danger })
  if (open) lastContent.current = { title, message, danger }
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    confirmRef.current?.focus()
    function handleKey(event: KeyboardEvent) {
      if (event.key === 'Escape') cancel.current()
      if (event.key === 'Tab') {
        const items = dialogRef.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')
        if (!items?.length) return
        const first = items[0], last = items[items.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => { window.removeEventListener('keydown', handleKey); previous?.focus() }
  }, [open])
  if (!present) return null

  return createPortal(
    <div className={`confirm-overlay ${open ? 'is-open' : 'is-closing'}`} inert={!open} onClick={onCancel}>
      <div ref={dialogRef} aria-labelledby={titleId} className="confirm-dialog" onClick={(e) => e.stopPropagation()} role="alertdialog" aria-modal="true">
        <h3 id={titleId} className="confirm-title">{lastContent.current.title}</h3>
        <p className="confirm-message">{lastContent.current.message}</p>
        <div className="confirm-actions">
          <button className="confirm-btn cancel" type="button" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            className={lastContent.current.danger ? 'confirm-btn danger' : 'confirm-btn primary'}
            type="button"
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>, document.body
  )
}

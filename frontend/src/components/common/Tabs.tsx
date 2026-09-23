import { useLayoutEffect, useRef, useState } from 'react'

export function Tabs<T extends string>({
  value,
  onChange,
  items,
  label,
}: {
  value: T
  onChange: (value: T) => void
  items: { value: T; label: string }[]
  label: string
}) {
  const root = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState({ left: 4, width: 0 })
  useLayoutEffect(() => {
    const update = () => {
      const selected = root.current?.querySelector<HTMLElement>('[aria-selected="true"]')
      if (selected) setPosition({ left: selected.offsetLeft, width: selected.offsetWidth })
    }
    update()
    const observer = new ResizeObserver(update)
    if (root.current) observer.observe(root.current)
    return () => observer.disconnect()
  }, [value])
  return (
    <div ref={root} className="sliding-tabs" role="tablist" aria-label={label}>
      <span
        className="tab-indicator"
        style={{ width: position.width, transform: `translateX(${position.left}px)` }}
      />
      {items.map((item, index) => (
        <button
          role="tab"
          aria-selected={item.value === value}
          tabIndex={item.value === value ? 0 : -1}
          key={item.value}
          onClick={() => onChange(item.value)}
          onKeyDown={(event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
            event.preventDefault()
            const next =
              event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? items.length - 1
                  : (index + (event.key === 'ArrowRight' ? 1 : -1) + items.length) % items.length
            onChange(items[next]!.value)
            root.current?.querySelectorAll<HTMLButtonElement>('button')[next]?.focus()
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}

import { useEffect, useLayoutEffect, useRef, useState } from 'react'

export function usePresence(open: boolean, duration = 160) {
  const [present, setPresent] = useState(open)
  useEffect(() => {
    if (open) {
      setPresent(true)
      return
    }
    const timer = window.setTimeout(
      () => setPresent(false),
      window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : duration,
    )
    return () => window.clearTimeout(timer)
  }, [open, duration])
  return open || present
}

/** Stable item identities prevent polling and refreshes from replaying entrances. */
export function useListMotion() {
  const root = useRef<HTMLDivElement>(null)
  const seen = useRef(new Set<string>())
  useLayoutEffect(() => {
    let index = 0
    root.current?.querySelectorAll<HTMLElement>('[data-entry]').forEach((item) => {
      const id = item.dataset.entry!
      if (seen.current.has(id)) return
      seen.current.add(id)
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || index >= 6) return
      item.animate(
        [
          { opacity: 0, transform: 'translateY(8px)' },
          { opacity: 1, transform: 'translateY(0)' },
        ],
        {
          duration: 200,
          delay: index++ * 25,
          easing: 'cubic-bezier(.22,1,.36,1)',
          fill: 'backwards',
        },
      )
    })
  })
  return root
}

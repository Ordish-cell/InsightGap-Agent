import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import * as feed from '../api/feed'
import type { FeedCard } from '../api/types'
import { Icon } from '../components/common/Icon'
import { ActionNotice } from '../components/common/ActionNotice'
import { AgentChatPanel } from '../components/agent/AgentChatPanel'

// Module-level cache to survive component remounts during hot reload / StrictMode
let _cachedCards: FeedCard[] | null = null
let _cacheTs = 0
const CACHE_TTL_MS = 30_000

type PickedFeed = { label: string; percent: string; className: string; card: FeedCard }

const relationMeta: Record<string, { label: string; percent: string; className: string }> = {
  explicit_related: { label: '显性相关', percent: '30%', className: 'explicit' },
  adjacent_domain: { label: '邻近机会', percent: '40%', className: 'adjacent' },
  far_domain: { label: '远域启发', percent: '30%', className: 'far' },
}

function relationOf(card: FeedCard | null | undefined): string {
  if (!card) return ''
  return card?.exposure_bucket || card?.relation_type || ''
}

function pickHomeFeeds(rawCards: FeedCard[]): PickedFeed[] {
  // Defensive: filter out falsy cards
  const safeCards = Array.isArray(rawCards) ? rawCards.filter(Boolean) : []
  const picked: PickedFeed[] = []

  for (const key of ['explicit_related', 'adjacent_domain', 'far_domain']) {
    const meta = relationMeta[key]
    if (!meta) continue
    const match = safeCards.find((card) => relationOf(card) === key)
    if (match) picked.push({ ...meta, card: match })
  }

  if (picked.length < 3 && safeCards.length > 0) {
    const found = picked.map((p) => relationOf(p?.card))
    const missing = ['explicit_related', 'adjacent_domain', 'far_domain'].filter((k) => !found.includes(k))
    console.warn(`[feed] home missing buckets: ${missing.join(', ')} (showing ${picked.length}/3)`)
  }

  return picked
}

function getInitialFeedOpen() {
  try {
    const saved = localStorage.getItem('homeFeedOpen')
    if (saved === 'false') return false
    if (saved === 'true') return true
  } catch { /* localStorage not available */ }
  return false
}

export function HomePage() {
  const researchLock = useRef(false)
  const navigate = useNavigate()
  const [cards, setCards] = useState<FeedCard[]>([])
  const [feedLoading, setFeedLoading] = useState(true)
  const [feedError, setFeedError] = useState('')
  const [feedOpen, setFeedOpen] = useState(getInitialFeedOpen)
  const [selectedFeedCardId, setSelectedFeedCardId] = useState<number | null>(null)
  const [researchingCardId, setResearchingCardId] = useState<number | null>(null)

  const homeFeeds = useMemo(() => pickHomeFeeds(cards), [cards])
  const firstHomeFeed = homeFeeds[0]
  const selectedFeedCard = useMemo(
    () => cards.find((card) => card?.id === selectedFeedCardId),
    [cards, selectedFeedCardId],
  )

  useEffect(() => {
    // Use cached data immediately to avoid loading flash on remount
    if (_cachedCards && Date.now() - _cacheTs < CACHE_TTL_MS) {
      setCards(_cachedCards)
      setFeedLoading(false)
    } else {
      setFeedLoading(true)
    }
    setFeedError('')
    try {
      feed.homeCards()
        .then((result) => {
          const safeCards: FeedCard[] = Array.isArray(result?.cards) ? result.cards.filter(Boolean) : []
          _cachedCards = safeCards
          _cacheTs = Date.now()
          setCards(safeCards)

          const refreshResult = result?.refresh_result as Record<string, unknown> | undefined
          if (refreshResult?.refreshed) {
            const missing = refreshResult.missing_buckets as string[] | undefined
            if (missing && missing.length > 0) {
              console.warn('[feed] missing buckets after refresh:', missing.join(', '))
            }
            if (!refreshResult.is_complete) {
              console.warn('[feed] batch incomplete, using available cards')
            }
            const sourceSummary = refreshResult.source_summary as Record<string, { search_count: number; seed_count: number; providers: string[] }> | undefined
            if (sourceSummary) {
              for (const [bucket, info] of Object.entries(sourceSummary)) {
                if (info?.seed_count > 0) {
                  console.warn(`[feed] bucket ${bucket} used ${info.seed_count} seed(s) as fallback (real search: ${info.search_count})`)
                }
                if (info?.search_count === 0 && (info?.providers?.length ?? 0) > 0) {
                  console.warn(`[feed] bucket ${bucket} has zero real search results, providers: ${info.providers.join(', ')}`)
                }
              }
            }
          }
        })
        .catch((exc) => {
          setCards([])
          setFeedError(exc instanceof Error ? exc.message : String(exc))
          console.warn('[feed] home cards load failed:', exc instanceof Error ? exc.message : String(exc))
        })
        .finally(() => setFeedLoading(false))
    } catch (exc) {
      setCards([])
      setFeedError(exc instanceof Error ? exc.message : String(exc))
      setFeedLoading(false)
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem('homeFeedOpen', String(feedOpen))
    } catch { /* ignore */ }
  }, [feedOpen])

  async function startResearch(cardId: number) {
    if (researchLock.current) return
    researchLock.current = true
    setResearchingCardId(cardId)
    setFeedError('')
    try {
      const result = await feed.startResearch(cardId)
      if (result?.id) {
        navigate(`/research/${result.id}`)
      } else {
        throw new Error('研究任务创建成功但没有返回 run_id')
      }
    } catch (err) {
      setFeedError(err instanceof Error ? err.message : '研究创建失败，请重试。')
    } finally {
      researchLock.current = false
      setResearchingCardId(null)
    }
  }

  return (
    <section className="home-page">
      <header className="home-topbar"><span className="home-context"><Icon name="chat" size={17} />对话工作台</span><button className={`feed-toggle ${feedOpen ? 'active' : ''}`} aria-expanded={feedOpen} onClick={() => setFeedOpen(value => !value)}><Icon name="feed" size={17} />今日精选<span className="count-label">{homeFeeds.length}</span><Icon name="chevron" size={14} style={{ transform: feedOpen ? 'rotate(90deg)' : undefined }} /></button></header>
      {!feedOpen && firstHomeFeed ? <button className="home-feed-peek" type="button" onClick={() => setFeedOpen(true)} aria-label={`展开今日信号：${firstHomeFeed.card.display_title || firstHomeFeed.card.title}`}><span className="home-feed-peek-label"><Icon name="spark" size={16} />今日信号</span><strong>{firstHomeFeed.card.display_title || firstHomeFeed.card.title}</strong><span className="home-feed-peek-action">展开精选 <Icon name="arrow" size={16} /></span></button> : null}
      <div className={`home-feed-disclosure ${feedOpen ? 'is-open' : ''}`} inert={!feedOpen}>
        <div><section className="home-feed-content" aria-label="今日精选">
          <div className="section-title"><div><strong>值得关注的信号</strong><p className="muted small">从一条信息开始，形成自己的判断。</p></div><Link className="text-link" to="/feed">全部信息流 <Icon name="arrow" size={16} /></Link></div>
          <ActionNotice message={feedError} error />
          {feedLoading ? <div className="feed-skeleton" aria-label="正在加载精选"><span /><span /><span /></div> : !homeFeeds.length ? <p className="muted">暂无精选信息，可以前往信息流刷新。</p> : <div className="floating-feed-grid">{homeFeeds.map(({ card: c, label, className }, index) => <article className={`floating-feed-card ${className} ${selectedFeedCardId === Number(c.id) ? 'is-selected' : ''}`} key={c.id}>
            <div className="floating-feed-card-top"><span className="eyebrow">{label}</span><span className="floating-feed-card-index">{String(index + 1).padStart(2, '0')}</span></div>
            <Link to={`/feed/${c.id}`} className="feed-title-link"><h3>{c.display_title || c.title || '未命名卡片'}</h3></Link>
            <p>{c.one_sentence_value || c.summary || '暂无摘要。'}</p>
            <div className="floating-feed-card-source">{c.domain || '来源未标记'}</div>
            <div className="floating-feed-card-actions"><button className={`button small ${selectedFeedCardId === Number(c.id) ? '' : 'secondary'}`} aria-pressed={selectedFeedCardId === Number(c.id)} onClick={() => setSelectedFeedCardId(current => current === Number(c.id) ? null : Number(c.id))}>{selectedFeedCardId === Number(c.id) ? '取消带入' : '带入对话'}</button><button className="button ghost small" onClick={() => void startResearch(Number(c.id))} disabled={researchingCardId !== null}>{researchingCardId === Number(c.id) ? '创建中…' : '深度研究'}<Icon name="arrow" size={14} /></button></div>
          </article>)}</div>}
        </section></div>
      </div>
      {selectedFeedCard && <div className="selected-feed-context" role="status"><Icon name="feed" size={16} /><span>围绕「{selectedFeedCard.title}」对话</span><button className="icon-button" aria-label="取消带入卡片" onClick={() => setSelectedFeedCardId(null)}><Icon name="close" size={16} /></button></div>}
      <AgentChatPanel source="home_chat" pageContext={{ page: 'home', selected_feed_card_id: selectedFeedCardId, selected_feed_card_title: selectedFeedCard?.title || '' }} placeholder={selectedFeedCard ? `围绕这张卡片提问：${selectedFeedCard.title}` : '提出问题，或上传文件一起探索…'} initialTitle="今天，想探索什么？" locale="zh" />
    </section>
  )
}

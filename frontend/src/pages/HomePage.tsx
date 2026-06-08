import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import * as feed from '../api/feed'
import type { FeedCard } from '../api/types'
import { AgentChatPanel } from '../components/agent/AgentChatPanel'

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
  return true
}

export function HomePage() {
  const navigate = useNavigate()
  const [cards, setCards] = useState<FeedCard[]>([])
  const [feedLoading, setFeedLoading] = useState(true)
  const [feedError, setFeedError] = useState('')
  const [feedOpen, setFeedOpen] = useState(getInitialFeedOpen)
  const [selectedFeedCardId, setSelectedFeedCardId] = useState<number | null>(null)

  const homeFeeds = useMemo(() => pickHomeFeeds(cards), [cards])
  const selectedFeedCard = useMemo(
    () => cards.find((card) => card?.id === selectedFeedCardId),
    [cards, selectedFeedCardId],
  )

  useEffect(() => {
    setFeedLoading(true)
    setFeedError('')
    try {
      feed.homeCards()
        .then((result) => {
          const safeCards: FeedCard[] = Array.isArray(result?.cards) ? result.cards.filter(Boolean) : []
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
    try {
      const result = (await feed.startResearch(cardId)) as { id?: string }
      if (result?.id) navigate(`/research/${result.id}`)
    } catch { /* navigation will handle it */ }
  }

  const isLoading = feedLoading
  const hasCards = homeFeeds.length > 0
  const hasError = !!feedError && !hasCards

  return (
    <section className="home-page">
      <div className={feedOpen ? 'floating-feed open' : 'floating-feed closed'}>
        <div className="floating-feed-head">
          <div>
            <h2>今日精选信息差</h2>
            <p>按显性相关、邻近机会和远域启发混合呈现的三条信号。</p>
          </div>
          <div className="floating-feed-actions">
            <Link className="soft-button" to="/feed">
              完整信息流
            </Link>
            <button className="soft-button ghost" onClick={() => setFeedOpen((value) => !value)}>
              {feedOpen ? '收起' : '展开'}
            </button>
          </div>
        </div>
        {feedOpen ? (
          <div className="floating-feed-grid">
            {isLoading ? (
              <article className="floating-feed-card explicit">
                <h3>正在加载真实信息差</h3>
                <p className="feed-value-line">正在读取数据库并按配置尝试刷新真实来源。</p>
              </article>
            ) : null}
            {!isLoading && hasError ? (
              <article className="floating-feed-card explicit">
                <h3>加载失败</h3>
                <p className="feed-value-line">{feedError || '未知错误'}</p>
                <Link className="light-mini-button" to="/feed">去信息流查看</Link>
              </article>
            ) : null}
            {!isLoading && !hasError && !hasCards ? (
              <article className="floating-feed-card explicit">
                <h3>今日信息正在生成</h3>
                <p className="feed-value-line">系统正在整理今日信息差，请稍后刷新或点击下方按钮重试。</p>
                <Link className="light-mini-button" to="/feed">去信息流手动刷新</Link>
              </article>
            ) : null}
            {homeFeeds.map((item, index) => {
              const c = item?.card
              if (!c) return null
              return (
                <article className={`floating-feed-card ${item.className}`} style={{ animationDelay: `${index * 140}ms` }} key={`${item.label}-${c.id || index}`}>
                  <div className="floating-feed-card-top">
                    <span className="mix-pill">{item.percent}</span>
                    <span className="mini-pill">{item.label}</span>
                    <span className="mini-pill score">分数：{Math.round((c.final_score ?? 0) * 100)}</span>
                  </div>
                  <h3 title={c.original_title || c.display_title || c.title || ''}>
                    {c.display_title || c.title || '未命名卡片'}
                  </h3>
                  <p className="feed-value-line">{c.one_sentence_value || c.summary || '这条信号可能带来新的判断角度。'}</p>
                  <div className="feed-relevance-line">
                    <strong>相关</strong>
                    <span>{c.why_relevant || c.why_you || '与你当前关注的方向有交集。'}</span>
                  </div>
                  <div className="feed-benefit-line">
                    <strong>好处</strong>
                    <span>{c.benefit || '可能对你的产品和技术决策有参考价值。'}</span>
                  </div>
                  <div className="mini-insight">
                    <strong>信息差</strong>
                    <span>{c.information_gap || '暂无明确信息差说明，可由后续研究补全。'}</span>
                  </div>
                  <div className="floating-feed-card-actions">
                    <button className="light-mini-button" onClick={() => setSelectedFeedCardId(c.id as unknown as number)}>
                      带入对话
                    </button>
                    <button className="dark-mini-button" onClick={() => startResearch(c.id as unknown as number)}>
                      深度研究
                    </button>
                    <Link className="light-mini-button" to={`/feed/${c.id}`}>
                      详情
                    </Link>
                  </div>
                </article>
              )
            })}
          </div>
        ) : null}
      </div>
      {!feedOpen ? (
        <button className="floating-feed-reopen" onClick={() => setFeedOpen(true)}>
          今日精选
        </button>
      ) : null}
      <AgentChatPanel
        source="home_chat"
        pageContext={{ page: 'home', selected_feed_card_id: selectedFeedCardId, selected_feed_card_title: selectedFeedCard?.title || '' }}
        placeholder={selectedFeedCard ? `围绕这张卡片提问：${selectedFeedCard.title}` : '让 Agent 帮你研究、生成成果、总结信息，或沉淀成可复用 Skill'}
        initialTitle="我们该构建或研究什么？"
        locale="zh"
      />
    </section>
  )
}

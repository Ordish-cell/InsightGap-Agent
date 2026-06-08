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

function relationOf(card: FeedCard) {
  return card.exposure_bucket || card.relation_type || ''
}

function pickHomeFeeds(cards: FeedCard[]): PickedFeed[] {
  const picked: PickedFeed[] = []
  const usedIds = new Set<number>()
  const usedTitles = new Set<string>()
  const usedUrls = new Set<string>()

  function tryAdd(card: FeedCard, meta: { label: string; percent: string; className: string }): boolean {
    if (usedIds.has(Number(card.id))) return false
    const titleKey = (card.original_title || card.display_title || card.title || '').toLowerCase().slice(0, 80)
    if (titleKey && usedTitles.has(titleKey)) return false
    const sourceUrl = (card as Record<string, unknown>).source_url as string
    if (sourceUrl && usedUrls.has(sourceUrl.toLowerCase())) return false
    usedIds.add(Number(card.id))
    if (titleKey) usedTitles.add(titleKey)
    if (sourceUrl) usedUrls.add(sourceUrl.toLowerCase())
    picked.push({ ...meta, card })
    return true
  }

  // First pass: pick one per bucket
  for (const key of ['explicit_related', 'adjacent_domain', 'far_domain']) {
    const meta = relationMeta[key]
    if (!meta) continue
    const match = cards.find((card) => relationOf(card) === key)
    if (match) tryAdd(match, meta)
  }

  // Second pass: fill remaining slots from any bucket (no duplicates)
  for (const card of cards) {
    if (picked.length >= 3) break
    const rel = relationOf(card)
    const meta = relationMeta[rel] || relationMeta.explicit_related
    tryAdd(card, meta)
  }

  return picked.slice(0, 3)
}

function getInitialFeedOpen() {
  const saved = localStorage.getItem('homeFeedOpen')
  if (saved === 'false') return false
  if (saved === 'true') return true
  localStorage.setItem('homeFeedOpen', 'true')
  return true
}

export function HomePage() {
  const navigate = useNavigate()
  const [cards, setCards] = useState<FeedCard[]>([])
  const [feedLoading, setFeedLoading] = useState(true)
  const [feedOpen, setFeedOpen] = useState(getInitialFeedOpen)
  const [selectedFeedCardId, setSelectedFeedCardId] = useState<number | null>(null)

  const homeFeeds = useMemo(() => pickHomeFeeds(cards), [cards])
  const selectedFeedCard = useMemo(() => cards.find((card) => card.id === selectedFeedCardId), [cards, selectedFeedCardId])

  useEffect(() => {
    setFeedLoading(true)
    feed.homeCards()
      .then((result) => {
        setCards(result.cards || [])
        const refreshResult = result.refresh_result as Record<string, unknown> | undefined
        if (refreshResult?.refreshed) {
          const missing = refreshResult.missing_buckets as string[] | undefined
          if (missing && missing.length > 0) {
            console.warn('[feed] missing buckets after refresh:', missing.join(', '))
          }
          if (!refreshResult.is_complete) {
            console.warn('[feed] batch incomplete, using available cards')
          }
        }
      })
      .catch((exc) => {
        setCards([])
        console.warn('[feed] home cards load failed:', exc instanceof Error ? exc.message : String(exc))
      })
      .finally(() => setFeedLoading(false))
  }, [])
  useEffect(() => {
    localStorage.setItem('homeFeedOpen', String(feedOpen))
  }, [feedOpen])

  async function startResearch(cardId: number) {
    const result = (await feed.startResearch(cardId)) as { id?: string }
    if (result?.id) navigate(`/research/${result.id}`)
  }

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
            {feedLoading ? (
              <article className="floating-feed-card explicit">
                <h3>正在加载真实信息差</h3>
                <p className="feed-value-line">正在读取数据库并按配置尝试刷新真实来源。</p>
              </article>
            ) : null}
            {!feedLoading && !homeFeeds.length ? (
              <article className="floating-feed-card explicit">
                <h3>暂无信息差卡片</h3>
                <p className="feed-value-line">请前往信息流页面手动刷新。</p>
                <Link className="light-mini-button" to="/feed">去信息流查看</Link>
              </article>
            ) : null}
            {homeFeeds.map((item, index) => (
              <article className={`floating-feed-card ${item.className}`} style={{ animationDelay: `${index * 140}ms` }} key={`${item.label}-${item.card.id}`}>
                <div className="floating-feed-card-top">
                  <span className="mix-pill">{item.percent}</span>
                  <span className="mini-pill">{item.label}</span>
                  <span className="mini-pill score">分数：{Math.round((item.card.final_score || 0) * 100)}</span>
                </div>
                <h3 title={item.card.original_title || item.card.display_title || item.card.title}>{item.card.display_title || item.card.title}</h3>
                <p className="feed-value-line">{item.card.one_sentence_value || item.card.summary || '这条信号可能带来新的判断角度。'}</p>
                <div className="feed-relevance-line">
                  <strong>相关</strong>
                  <span>{item.card.why_relevant || item.card.why_you || '与你当前关注的方向有交集。'}</span>
                </div>
                <div className="feed-benefit-line">
                  <strong>好处</strong>
                  <span>{item.card.benefit || '可能对你的产品和技术决策有参考价值。'}</span>
                </div>
                <div className="mini-insight">
                  <strong>信息差</strong>
                  <span>{item.card.information_gap || '暂无明确信息差说明，可由后续研究补全。'}</span>
                </div>
                <div className="floating-feed-card-actions">
                  <button className="light-mini-button" onClick={() => setSelectedFeedCardId(item.card.id)}>
                    带入对话
                  </button>
                  <button className="dark-mini-button" onClick={() => startResearch(item.card.id)}>
                    深度研究
                  </button>
                  <Link className="light-mini-button" to={`/feed/${item.card.id}`}>
                    详情
                  </Link>
                </div>
              </article>
            ))}
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

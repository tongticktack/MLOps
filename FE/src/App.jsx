import { useEffect, useMemo, useRef, useState } from 'react'

const API_URL = 'http://localhost:7070/predict'
const WS_URL = 'ws://localhost:7070/ws/mlops'
const MAX_POINTS = 96

const formatNumber = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 }).format(value)
}

const formatDateTime = (value) => {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return `${parsed.getMonth() + 1}/${parsed.getDate()} ${String(parsed.getHours()).padStart(2, '0')}:${String(parsed.getMinutes()).padStart(2, '0')}`
}

const buildLinePath = (points, width, height, accessor, minValue, maxValue) => {
  if (!points.length) return ''
  const range = Math.max(maxValue - minValue, 1)

  return points
    .map((point, index) => {
      const x = points.length === 1 ? width / 2 : (index / (points.length - 1)) * width
      const y = height - ((accessor(point) - minValue) / range) * height
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

export default function App() {
  const socketRef = useRef(null)
  const isPausedRef = useRef(false)
  const messageBufferRef = useRef([])

  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [streamStatus, setStreamStatus] = useState('Idle')
  const [apiError, setApiError] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [rmse, setRmse] = useState('-')
  const [rmse24h, setRmse24h] = useState('-')
  const [rmse168h, setRmse168h] = useState('-')
  const [threshold, setThreshold] = useState('-')
  const [baselineRmse, setBaselineRmse] = useState('-')
  const [recordCount, setRecordCount] = useState(0)
  const [currentPredictionTime, setCurrentPredictionTime] = useState('')
  const [retrainCount, setRetrainCount] = useState(0)
  const [latestRetrainReason, setLatestRetrainReason] = useState('')
  const [lastRetrainAt, setLastRetrainAt] = useState('')
  const [latestReport, setLatestReport] = useState('')
  const [typedReport, setTypedReport] = useState('')
  const [streamPoints, setStreamPoints] = useState([])
  const [retrainRecordCounts, setRetrainRecordCounts] = useState([])
  const [retrainPopup, setRetrainPopup] = useState(null) // { trainFrom, trainTo, reason }

  useEffect(() => {
    return () => {
      if (socketRef.current) socketRef.current.close()
    }
  }, [])

  useEffect(() => {
    if (!latestReport) {
      setTypedReport('')
      return
    }

    let index = 0
    const timer = window.setInterval(() => {
      index += 1
      setTypedReport(latestReport.slice(0, index))
      if (index >= latestReport.length) window.clearInterval(timer)
    }, 12)

    return () => window.clearInterval(timer)
  }, [latestReport])

  const fileSummary = useMemo(() => {
    if (!selectedFile) return '업로드된 파일이 없습니다.'
    const sizeInMb = (selectedFile.size / (1024 * 1024)).toFixed(2)
    return `${selectedFile.name} · ${sizeInMb} MB`
  }, [selectedFile])

  const chartMetrics = useMemo(() => {
    if (!streamPoints.length) {
      return {
        actualPath: '',
        predictedPath: '',
        minValue: 0,
        maxValue: 0,
        width: 1100,
        height: 360,
      }
    }

    const width = 1100
    const height = 360
    const values = streamPoints.flatMap((point) => [point.y, point.y_hat])
    const minValue = Math.min(...values)
    const maxValue = Math.max(...values)

    return {
      actualPath: buildLinePath(streamPoints, width, height, (point) => point.y, minValue, maxValue),
      predictedPath: buildLinePath(streamPoints, width, height, (point) => point.y_hat, minValue, maxValue),
      minValue,
      maxValue,
      width,
      height,
    }
  }, [streamPoints])

  const errorChart = useMemo(() => {
    if (!streamPoints.length) {
      return { width: 1100, height: 120, bars: [], retrainMarkers: [] }
    }

    const width = 1100
    const height = 120
    const maxError = Math.max(...streamPoints.map((point) => point.error || 0), 1)
    const barWidth = width / Math.max(streamPoints.length, 1)

    const firstCount = streamPoints[0]?.record_count ?? 0
    const lastCount = streamPoints[streamPoints.length - 1]?.record_count ?? 0
    const countRange = Math.max(lastCount - firstCount, 1)

    const retrainMarkers = retrainRecordCounts
      .filter((rc) => rc >= firstCount && rc <= lastCount)
      .map((rc) => ((rc - firstCount) / countRange) * width)

    return {
      width,
      height,
      bars: streamPoints.map((point, index) => ({
        x: index * barWidth,
        width: Math.max(barWidth - 2, 2),
        height: ((point.error || 0) / maxError) * height,
      })),
      retrainMarkers,
    }
  }, [streamPoints, retrainRecordCounts])

  const latestPoint = streamPoints.at(-1)
  const statusTone =
    streamStatus === 'drift' || streamStatus === 'Error'
      ? 'border-rose-400/30 bg-rose-400/10 text-rose-200'
      : streamStatus === 'warning' || streamStatus === 'Retraining'
        ? 'border-amber-300/30 bg-amber-300/10 text-amber-200'
        : streamStatus === 'Streaming' || streamStatus === 'normal' || streamStatus === 'Completed'
          ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
          : streamStatus === 'Stopped'
            ? 'border-rose-400/30 bg-rose-400/10 text-rose-300'
            : 'border-slate-400/20 bg-slate-400/10 text-slate-300'

  const closeRetrainPopup = () => {
    setRetrainPopup(null)
    isPausedRef.current = false
    const buffered = messageBufferRef.current
    messageBufferRef.current = []
    const processMessage = socketRef.current?._processMessage
    if (processMessage) buffered.forEach(processMessage)
  }

  const isStreaming = ['Streaming', 'Retraining', 'Warming Up', 'warning', 'drift', 'normal'].includes(streamStatus)

  const stopStream = () => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    setStreamStatus('Stopped')
    setIsLoading(false)
  }

  const resetStream = () => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    isPausedRef.current = false
    messageBufferRef.current = []

    setStreamStatus('Idle')
    setApiError('')
    setSessionId('')
    setRmse('-')
    setRmse24h('-')
    setRmse168h('-')
    setThreshold('-')
    setBaselineRmse('-')
    setRecordCount(0)
    setCurrentPredictionTime('')
    setRetrainCount(0)
    setLatestRetrainReason('')
    setLastRetrainAt('')
    setLatestReport('')
    setTypedReport('')
    setStreamPoints([])
    setRetrainRecordCounts([])
  }

  const handleFile = (file) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setApiError('CSV 파일만 업로드할 수 있습니다.')
      return
    }

    setSelectedFile(file)
    resetStream()
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  const openStream = (sessionIdValue) => {
    if (socketRef.current) socketRef.current.close()

    const socket = new WebSocket(`${WS_URL}?session_id=${sessionIdValue}`)
    socketRef.current = socket

    socket.onopen = () => {
      setStreamStatus('Streaming')
      setIsLoading(false)
    }

    const processMessage = (payload) => {
      if (payload?.error && typeof payload.error === 'object' && !Array.isArray(payload.error)) {
        setApiError(payload.error.message || '스트림 처리 중 오류가 발생했습니다.')
        setStreamStatus('Error')
        return
      }

      if (payload?.event === 'stream_complete') {
        setStreamStatus('Completed')
        if (typeof payload.retrain_count === 'number') setRetrainCount(payload.retrain_count)
        return
      }

      const point = {
        timestamp: payload.timestamp,
        y: payload.y,
        y_hat: payload.y_hat,
        error: payload.error,
        rmse: payload.rmse,
        rmse_24h: payload.rmse_24h,
        rmse_168h: payload.rmse_168h,
        record_count: payload.record_count,
        current_prediction_time: payload.current_prediction_time,
        retrain: payload.retrain,
        retrain_reason: payload.retrain_reason,
        threshold: payload.threshold,
        baseline_rmse: payload.baseline_rmse,
        llm_report: payload.llm_report,
      }

      setStreamPoints((current) => [...current.slice(-(MAX_POINTS - 1)), point])
      setRmse(typeof payload.rmse === 'number' ? `${payload.rmse}%` : String(payload.rmse ?? '-'))
      setRmse24h(payload.rmse_24h === null || payload.rmse_24h === undefined ? '-' : `${payload.rmse_24h} MWh`)
      setRmse168h(payload.rmse_168h === null || payload.rmse_168h === undefined ? '-' : `${payload.rmse_168h} MWh`)
      setThreshold(payload.threshold === null || payload.threshold === undefined ? '-' : `${payload.threshold} MWh`)
      setBaselineRmse(payload.baseline_rmse === null || payload.baseline_rmse === undefined ? '-' : `${payload.baseline_rmse} MWh`)
      setRecordCount(payload.record_count ?? 0)
      setCurrentPredictionTime(payload.current_prediction_time ?? '')
      setLatestReport(payload.llm_report ?? '')
      setStreamStatus(
        payload.pipeline_status === 'retraining'
          ? 'Retraining'
          : payload.pipeline_status === 'warming_up'
            ? 'Warming Up'
            : payload.pipeline_status === 'warning'
              ? 'warning'
              : payload.pipeline_status === 'drift'
                ? 'drift'
                : 'normal'
      )

      if (payload.retrain) {
        setRetrainCount((current) => current + 1)
        setLatestRetrainReason(payload.retrain_reason ?? 'Threshold exceeded')
        setLastRetrainAt(payload.current_prediction_time ?? payload.timestamp ?? '')
        setRetrainRecordCounts((current) => [...current, payload.record_count])
        isPausedRef.current = true
        setRetrainPopup({
          trainFrom: payload.train_from ?? '-',
          trainTo: payload.train_to ?? '-',
          reason: payload.retrain_reason ?? 'RMSE_168h > threshold',
        })
      }
    }

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data)
      if (isPausedRef.current) {
        messageBufferRef.current.push(payload)
        return
      }
      processMessage(payload)
    }

    socket._processMessage = processMessage

    socket.onerror = () => {
      setApiError('WebSocket 연결에 실패했습니다.')
      setStreamStatus('Error')
      setIsLoading(false)
    }

    socket.onclose = () => {
      setIsLoading(false)
      setStreamStatus((current) => (current === 'Completed' ? current : current === 'Error' ? current : 'Disconnected'))
    }
  }

  const submitPrediction = async () => {
    if (!selectedFile || isLoading) return

    setIsLoading(true)
    setApiError('')
    setStreamStatus('Starting')
    setStreamPoints([])
    setRetrainCount(0)
    setLatestRetrainReason('')
    setLatestReport('')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json().catch(() => ({}))

      if (!response.ok) {
        throw new Error(payload?.error?.message || '세션 생성에 실패했습니다.')
      }

      setSessionId(payload.session_id ?? '')
      setRecordCount(payload.record_count ?? 0)
      openStream(payload.session_id)
    } catch (error) {
      setApiError(error.message || '백엔드 연결에 실패했습니다.')
      setStreamStatus('Error')
      setIsLoading(false)
    }
  }

  return (
    <main className="relative overflow-hidden text-slate-100">
      {retrainPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="relative w-full max-w-md rounded-[2rem] border border-rose-400/30 bg-slate-950 p-8 shadow-2xl">
            <div className="mb-6 flex items-center gap-4">
              <span className="inline-flex h-12 w-12 animate-spin rounded-full border-4 border-rose-400 border-t-transparent" />
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-rose-400">MLOps Pipeline</p>
                <h2 className="font-orbitron text-2xl font-bold text-white">모델 재학습 중</h2>
              </div>
            </div>

            <div className="mb-6 space-y-3 rounded-[1.25rem] border border-white/10 bg-slate-900/80 p-5">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">학습 데이터 시작</span>
                <span className="font-mono font-semibold text-white">{retrainPopup.trainFrom}</span>
              </div>
              <div className="h-px bg-white/10" />
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">학습 데이터 종료</span>
                <span className="font-mono font-semibold text-white">{retrainPopup.trainTo}</span>
              </div>
              <div className="h-px bg-white/10" />
              <div className="flex items-start justify-between gap-4 text-sm">
                <span className="shrink-0 text-slate-400">트리거 사유</span>
                <span className="text-right text-rose-300">{retrainPopup.reason}</span>
              </div>
            </div>

            <p className="mb-6 text-center text-xs text-slate-500">
              재학습이 완료되었습니다. 확인 후 스트림을 재개하세요.
            </p>

            <button
              type="button"
              onClick={closeRetrainPopup}
              className="w-full rounded-full border border-spark-300/50 bg-gradient-to-r from-spark-400 to-spark-500 py-3 font-orbitron text-sm font-bold uppercase tracking-[0.2em] text-slate-950 transition hover:brightness-110"
            >
              확인 — 스트림 재개
            </button>
          </div>
        </div>
      )}
      <div className="pointer-events-none absolute inset-0 bg-grid bg-[size:42px_42px] opacity-10" />
      <div className="pointer-events-none absolute left-[-8rem] top-16 h-72 w-72 rounded-full bg-bolt-500/20 blur-3xl" />
      <div className="pointer-events-none absolute right-[-5rem] top-40 h-64 w-64 rounded-full bg-spark-400/20 blur-3xl" />

      <section className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-8 px-4 py-6 sm:px-6 lg:px-8">
        <header className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-neon backdrop-blur-xl">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-3 rounded-full border border-spark-300/30 bg-spark-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-spark-300">
              <span
                className={`h-2 w-2 rounded-full ${
                  streamStatus === 'Streaming' || streamStatus === 'Retraining'
                    ? 'animate-pulse bg-emerald-300'
                    : isLoading
                      ? 'animate-pulse bg-spark-300'
                      : 'bg-slate-400'
                }`}
              />
              Autonomous Grid Ops
            </div>
            <h1 className="font-orbitron text-4xl font-extrabold leading-tight text-white sm:text-5xl lg:text-6xl">
              AIOps for Power Demand
            </h1>
          </div>
        </header>

        <section className="grid gap-6">
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/60 p-6 shadow-neon backdrop-blur-xl">
            <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Live Forecast Stream</p>
                <h2 className="font-orbitron text-2xl font-bold text-white">실시간 예측 그래프</h2>
              </div>
              <div className="flex flex-wrap items-center gap-3">
              </div>
            </div>

            <div className="rounded-[1.5rem] border border-white/10 bg-gradient-to-br from-bolt-500/10 to-slate-900/80 p-5">
              {streamPoints.length ? (
                <div className="space-y-4">
                  <div className="grid gap-3 sm:grid-cols-4">
                    {[
                      ['Current Y', `${formatNumber(latestPoint?.y)}`],
                      ['Current Y_HAT', `${formatNumber(latestPoint?.y_hat)}`],
                      ['Current Error', `${formatNumber(latestPoint?.error)}`],
                      ['Prediction Time', formatDateTime(currentPredictionTime)],
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-2xl border border-white/10 bg-slate-950/55 px-4 py-3">
                        <p className="text-xs uppercase tracking-[0.22em] text-slate-500">{label}</p>
                        <p className="mt-2 font-orbitron text-lg font-bold text-white">{value}</p>
                      </div>
                    ))}
                  </div>

                  <div className="overflow-hidden rounded-[1.25rem] border border-white/10 bg-slate-950/70 p-4">
                    <svg viewBox={`0 0 ${chartMetrics.width} ${chartMetrics.height}`} className="h-[420px] w-full">
                      {[0.2, 0.4, 0.6, 0.8].map((ratio) => (
                        <line
                          key={ratio}
                          x1="0"
                          x2={chartMetrics.width}
                          y1={chartMetrics.height * ratio}
                          y2={chartMetrics.height * ratio}
                          stroke="rgba(255,255,255,0.08)"
                          strokeDasharray="6 6"
                        />
                      ))}

                      {streamPoints.map((point, index) => {
                        const range = Math.max(chartMetrics.maxValue - chartMetrics.minValue, 1)
                        const x = streamPoints.length === 1 ? chartMetrics.width / 2 : (index / (streamPoints.length - 1)) * chartMetrics.width
                        const yActual = chartMetrics.height - ((point.y - chartMetrics.minValue) / range) * chartMetrics.height
                        const yPredicted = chartMetrics.height - ((point.y_hat - chartMetrics.minValue) / range) * chartMetrics.height

                        return (
                          <g key={`${point.timestamp}-${index}`}>
                            <line x1={x} x2={x} y1={yActual} y2={yPredicted} stroke="rgba(250, 204, 21, 0.22)" strokeWidth="2" />
                            {point.retrain ? (
                              <>
                                <line x1={x} x2={x} y1="0" y2={chartMetrics.height} stroke="rgba(251, 113, 133, 0.75)" strokeDasharray="8 8" />
                                <circle cx={x} cy={yPredicted} r="7" fill="#fb7185" />
                              </>
                            ) : null}
                          </g>
                        )
                      })}

                      <path d={chartMetrics.actualPath} fill="none" stroke="#60a5fa" strokeWidth="4" strokeLinejoin="round" />
                      <path d={chartMetrics.predictedPath} fill="none" stroke="#facc15" strokeWidth="4" strokeDasharray="14 8" strokeLinejoin="round" />
                    </svg>
                  </div>

                  <div className="rounded-[1.25rem] border border-white/10 bg-slate-950/70 p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <p className="text-sm font-semibold text-white">Absolute Error Stream</p>
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-500">|y - y_hat|</p>
                    </div>
                    <svg viewBox={`0 0 ${errorChart.width} ${errorChart.height + 28}`} className="h-36 w-full">
                      {/* 막대 */}
                      {errorChart.bars.map((bar, index) => (
                        <rect
                          key={index}
                          x={bar.x}
                          y={errorChart.height + 28 - bar.height}
                          width={bar.width}
                          height={bar.height}
                          rx="3"
                          fill="#38bdf8"
                          opacity="0.75"
                        />
                      ))}
                      {/* retrain 마커 — record_count 기반으로 항상 추적 */}
                      {errorChart.retrainMarkers.map((x, i) => (
                        <g key={`rm-${i}`}>
                          <line
                            x1={x}
                            x2={x}
                            y1={0}
                            y2={errorChart.height + 28}
                            stroke="#fb7185"
                            strokeWidth="3"
                            strokeDasharray="6 4"
                          />
                          <rect x={x - 32} y={4} width="64" height="22" rx="5" fill="#fb7185" />
                          <text
                            x={x}
                            y={19}
                            textAnchor="middle"
                            fontSize="11"
                            fontWeight="bold"
                            fill="white"
                            fontFamily="monospace"
                          >
                            RETRAIN
                          </text>
                        </g>
                      ))}
                    </svg>
                  </div>

                  <div className="flex flex-wrap items-center gap-5 text-xs uppercase tracking-[0.22em] text-slate-400">
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-6 rounded-full bg-bolt-300" />
                      Actual y
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-6 rounded-full bg-spark-300" />
                      Predicted y_hat
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-6 rounded-full bg-rose-300" />
                      Retrain Trigger
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-6 rounded-full bg-white/50" />
                      Error Spread
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex h-[520px] flex-col items-center justify-center rounded-[1.25rem] border border-dashed border-white/10 bg-slate-950/70 p-8 text-center text-slate-400">
                  <svg viewBox="0 0 24 24" className="h-16 w-16 fill-current text-bolt-300">
                    <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
                  </svg>
                  <p className="mt-4 max-w-xl text-sm leading-7">
                    CSV 업로드 후 스트림을 시작하면 실제값과 예측값이 주식 차트처럼 연속 업데이트되고, 재학습이
                    발생한 시점은 세로 마커로 표시됩니다.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="grid gap-6">
          <div className="grid gap-6 lg:grid-cols-6">
            {[
              { label: 'RMSE 24H', value: rmse24h, tone: 'from-spark-400/15 to-slate-950/70' },
              { label: 'RMSE 168H', value: rmse168h, tone: 'from-bolt-500/15 to-slate-950/70' },
              { label: 'Threshold', value: threshold, tone: 'from-white/10 to-slate-950/70' },
              { label: 'Baseline RMSE', value: baselineRmse, tone: 'from-cyan-400/10 to-slate-950/70' },
              { label: 'Data Points', value: String(recordCount || 0), tone: 'from-emerald-400/10 to-slate-950/70' },
              { label: 'Retrain Count', value: String(retrainCount), tone: 'from-rose-400/10 to-slate-950/70' },
            ].map((item) => (
              <div
                key={item.label}
                className={`rounded-[2rem] border border-white/10 bg-gradient-to-br ${item.tone} p-5 shadow-neon backdrop-blur-xl`}
              >
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{item.label}</p>
                <p className="mt-3 font-orbitron text-2xl font-bold text-white sm:text-3xl">{item.value}</p>
              </div>
            ))}
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {[
              ['Pipeline', streamStatus],
              ['Current Time', formatDateTime(currentPredictionTime)],
              ['Active Dataset', selectedFile?.name ?? 'No file loaded'],
            ].map(([label, value]) => (
              <div key={label} className="rounded-[1.75rem] border border-white/10 bg-slate-950/60 p-5 backdrop-blur-xl">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">{label}</p>
                {label === 'Pipeline' ? (
                  <span className={`mt-3 inline-flex rounded-full border px-3 py-2 text-sm font-semibold uppercase tracking-[0.18em] ${statusTone}`}>
                    {value}
                  </span>
                ) : (
                  <p className="mt-3 font-semibold text-white">{value}</p>
                )}
              </div>
            ))}
          </div>

          <div className="rounded-[2rem] border border-bolt-300/20 bg-slate-950/60 p-6 shadow-neon backdrop-blur-xl">
            <div className="mb-6 flex items-center gap-3">
              <div className="rounded-2xl border border-spark-300/40 bg-spark-300/10 p-3 text-spark-300">
                <svg viewBox="0 0 24 24" className="h-7 w-7 fill-current">
                  <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
                </svg>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Stream Control Section</p>
                <h2 className="font-orbitron text-2xl font-bold text-white">운영 입력 및 재학습 임계치 제어</h2>
              </div>
            </div>

            <div className="grid gap-6">
              <div
                onDragOver={(event) => {
                  event.preventDefault()
                  setIsDragging(true)
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`group relative rounded-[1.75rem] border border-dashed p-6 transition duration-300 ${
                  isDragging
                    ? 'border-spark-300 bg-spark-300/10 shadow-pulse'
                    : 'border-bolt-300/30 bg-gradient-to-br from-bolt-500/10 to-slate-900/60'
                }`}
              >
                <div className="pointer-events-none absolute inset-0 rounded-[1.75rem] bg-gradient-to-br from-white/5 to-transparent" />
                <div className="relative flex flex-col items-center justify-center gap-5 py-10 text-center">
                  <div className="animate-float rounded-full border border-white/10 bg-white/5 p-5">
                    <svg viewBox="0 0 24 24" className="h-14 w-14 fill-current text-bolt-300">
                      <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-orbitron text-2xl font-bold text-white">Drag & Drop CSV</h3>
                    <p className="mt-2 text-sm text-slate-300">
                      `날짜,1시,...,24시` 포맷 CSV를 업로드하면 새 WebSocket 세션을 생성합니다.
                    </p>
                  </div>
                  <label className="cursor-pointer rounded-full border border-bolt-300/40 bg-bolt-500/20 px-5 py-3 text-sm font-semibold text-white transition hover:bg-bolt-500/30">
                    CSV 선택
                    <input
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={(event) => handleFile(event.target.files?.[0])}
                    />
                  </label>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm text-slate-300">
                    {fileSummary}
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-sm leading-6 text-slate-300">
                <p className="font-semibold text-white">실제 연동 대상</p>
                <p>`POST {API_URL}` 로 세션을 만들고, 이후 `WS {WS_URL}` 스트림을 실시간 수신합니다. 재학습 threshold는 백엔드 고정값을 사용합니다.</p>
              </div>
              <div className="flex gap-3">
                {isStreaming && (
                  <button
                    type="button"
                    onClick={stopStream}
                    className="inline-flex min-w-fit whitespace-nowrap items-center justify-center gap-2 rounded-full border border-rose-400/50 bg-gradient-to-r from-rose-500 to-rose-600 px-[clamp(1rem,3vw,1.5rem)] py-[clamp(0.75rem,2vw,0.95rem)] font-orbitron text-[clamp(0.7rem,1.8vw,0.875rem)] font-bold uppercase tracking-[clamp(0.08em,0.18vw,0.2em)] text-white transition hover:brightness-110"
                  >
                    <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current">
                      <rect x="4" y="4" width="16" height="16" rx="2" />
                    </svg>
                    스트림 중단
                  </button>
                )}
                <button
                  type="button"
                  onClick={submitPrediction}
                  disabled={!selectedFile || isLoading}
                  className="inline-flex min-w-fit whitespace-nowrap items-center justify-center gap-2 rounded-full border border-spark-300/50 bg-gradient-to-r from-spark-400 to-spark-500 px-[clamp(1rem,3vw,1.5rem)] py-[clamp(0.75rem,2vw,0.95rem)] font-orbitron text-[clamp(0.7rem,1.8vw,0.875rem)] font-bold uppercase tracking-[clamp(0.08em,0.18vw,0.2em)] text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isLoading ? (
                    <>
                      <span className="inline-flex h-5 w-5 animate-spin rounded-full border-2 border-slate-950 border-t-transparent" />
                      세션 생성 중
                    </>
                  ) : (
                    <>
                      <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current">
                        <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
                      </svg>
                      스트림 시작
                    </>
                  )}
                </button>
              </div>
            </div>

            {apiError ? (
              <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
                {apiError}
              </div>
            ) : null}

            <div className="mt-4 grid gap-3">
              <div className="rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm text-slate-300">
                Session ID: <span className="font-mono text-white">{sessionId || '-'}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm text-slate-300">
                Last Retrain: <span className="text-white">{lastRetrainAt ? formatDateTime(lastRetrainAt) : 'No retrain event yet'}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm text-slate-300">
                Retrain Reason: <span className="text-white">{latestRetrainReason || 'No retrain event yet'}</span>
              </div>
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-neon backdrop-blur-xl">
            <div className="mb-6 flex items-center gap-3">
              <div className="rounded-2xl border border-bolt-300/40 bg-bolt-400/10 p-3 text-bolt-200">
                <svg viewBox="0 0 24 24" className="h-7 w-7 fill-current">
                  <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
                </svg>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Live Report</p>
                <h2 className="font-orbitron text-2xl font-bold text-white">실시간 운영 보고서</h2>
              </div>
            </div>

            <div className="rounded-[1.75rem] border border-bolt-300/20 bg-slate-950/70 p-5">
              <div className="mb-4 flex items-center justify-between gap-4">
                <span className="rounded-full border border-spark-300/30 bg-spark-300/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-spark-300">
                  Stream Narrative
                </span>
                <span className="text-xs uppercase tracking-[0.25em] text-slate-500">{streamStatus}</span>
              </div>

              <div className="scrollbar-thin min-h-[320px] rounded-[1.25rem] border border-white/5 bg-gradient-to-br from-slate-900 to-slate-950 p-6 text-sm leading-8 text-slate-200">
                {typedReport ? (
                  <p className="whitespace-pre-wrap">
                    {typedReport}
                    <span className="ml-1 inline-block h-5 w-[2px] animate-pulse bg-spark-300 align-middle" />
                  </p>
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-4 px-4 text-center text-slate-400">
                    <svg viewBox="0 0 24 24" className="h-16 w-16 animate-flicker fill-current text-bolt-300">
                      <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
                    </svg>
                    <p>
                      스트림이 시작되면 최신 예측 오차와 재학습 발생 여부를 반영한 운영 보고서가 이 영역에
                      실시간으로 갱신됩니다.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}

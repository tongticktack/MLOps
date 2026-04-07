import { useEffect, useMemo, useRef, useState } from 'react'

const API_URL = 'http://localhost:7070/predict'
const WS_URL = 'ws://localhost:7070/ws/mlops'
const MAX_POINTS = 96

const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

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

  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [streamStatus, setStreamStatus] = useState('Idle')
  const [apiError, setApiError] = useState('')
  const [threshold, setThreshold] = useState(5)
  const [sessionId, setSessionId] = useState('')
  const [rmse, setRmse] = useState('-')
  const [recordCount, setRecordCount] = useState(0)
  const [currentPredictionTime, setCurrentPredictionTime] = useState('')
  const [retrainCount, setRetrainCount] = useState(0)
  const [latestRetrainReason, setLatestRetrainReason] = useState('')
  const [lastRetrainAt, setLastRetrainAt] = useState('')
  const [latestReport, setLatestReport] = useState('')
  const [typedReport, setTypedReport] = useState('')
  const [streamPoints, setStreamPoints] = useState([])

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

  const latestPoint = streamPoints.at(-1)

  const resetStream = () => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }

    setStreamStatus('Idle')
    setApiError('')
    setSessionId('')
    setRmse('-')
    setRecordCount(0)
    setCurrentPredictionTime('')
    setRetrainCount(0)
    setLatestRetrainReason('')
    setLastRetrainAt('')
    setLatestReport('')
    setTypedReport('')
    setStreamPoints([])
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

  const openStream = (sessionIdValue, thresholdValue) => {
    if (socketRef.current) socketRef.current.close()

    const socket = new WebSocket(`${WS_URL}?session_id=${sessionIdValue}&threshold=${thresholdValue}`)
    socketRef.current = socket

    socket.onopen = () => {
      setStreamStatus('Streaming')
      setIsLoading(false)
    }

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data)

      if (payload?.error) {
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
        record_count: payload.record_count,
        current_prediction_time: payload.current_prediction_time,
        retrain: payload.retrain,
        retrain_reason: payload.retrain_reason,
        threshold: payload.threshold,
        llm_report: payload.llm_report,
      }

      setStreamPoints((current) => [...current.slice(-(MAX_POINTS - 1)), point])
      setRmse(typeof payload.rmse === 'number' ? `${payload.rmse}%` : String(payload.rmse ?? '-'))
      setRecordCount(payload.record_count ?? 0)
      setCurrentPredictionTime(payload.current_prediction_time ?? '')
      setLatestReport(payload.llm_report ?? '')
      setStreamStatus(payload.pipeline_status === 'retraining' ? 'Retraining' : 'Streaming')

      if (payload.retrain) {
        setRetrainCount((current) => current + 1)
        setLatestRetrainReason(payload.retrain_reason ?? 'Threshold exceeded')
        setLastRetrainAt(payload.current_prediction_time ?? payload.timestamp ?? '')
      }
    }

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
      openStream(payload.session_id, threshold)
    } catch (error) {
      setApiError(error.message || '백엔드 연결에 실패했습니다.')
      setStreamStatus('Error')
      setIsLoading(false)
    }
  }

  return (
    <main className="relative overflow-hidden text-slate-100">
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
          <div className="grid gap-6 lg:grid-cols-5">
            {[
              { label: 'RMSE', value: rmse, tone: 'from-spark-400/15 to-slate-950/70' },
              { label: 'Threshold', value: `${threshold}%`, tone: 'from-bolt-500/15 to-slate-950/70' },
              { label: 'Data Points', value: String(recordCount || 0), tone: 'from-emerald-400/10 to-slate-950/70' },
              { label: 'Retrain Count', value: String(retrainCount), tone: 'from-rose-400/10 to-slate-950/70' },
              {
                label: 'Last Retrain',
                value: lastRetrainAt ? formatDateTime(lastRetrainAt) : 'No retrain',
                tone: 'from-orange-400/10 to-slate-950/70',
              },
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
                <p className="mt-3 font-semibold text-white">{value}</p>
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

            <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
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

              <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/55 p-5">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Retrain Threshold</p>
                <div className="mt-3 flex items-center gap-4">
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    value={threshold}
                    onChange={(event) => setThreshold(clamp(Number(event.target.value), 1, 20))}
                    className="w-full accent-yellow-400"
                  />
                  <input
                    type="number"
                    min="1"
                    max="20"
                    step="0.5"
                    value={threshold}
                    onChange={(event) => setThreshold(clamp(Number(event.target.value || 1), 1, 20))}
                    className="w-24 rounded-xl border border-white/10 bg-slate-900 px-3 py-2 text-right text-white"
                  />
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-300">
                  누적 RMSE가 임계치를 넘으면 재학습 이벤트를 표시합니다. 현재 세션은 `session_id`와 `threshold`
                  를 포함해 `ws://localhost:7070/ws/mlops`에 연결됩니다.
                </p>
                <div className="mt-5 grid gap-3">
                  <div className="rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm text-slate-300">
                    Session ID: <span className="font-mono text-white">{sessionId || '-'}</span>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm text-slate-300">
                    Last Retrain: <span className="text-white">{latestRetrainReason || 'No retrain event yet'}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-sm leading-6 text-slate-300">
                <p className="font-semibold text-white">실제 연동 대상</p>
                <p>`POST {API_URL}` 로 세션을 만들고, 이후 `WS {WS_URL}` 스트림을 실시간 수신합니다.</p>
              </div>
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

            {apiError ? (
              <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
                {apiError}
              </div>
            ) : null}
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

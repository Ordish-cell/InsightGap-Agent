export function JsonBlock({ value }: { value: unknown }) {
  return <pre className="code-block">{JSON.stringify(value ?? {}, null, 2)}</pre>
}

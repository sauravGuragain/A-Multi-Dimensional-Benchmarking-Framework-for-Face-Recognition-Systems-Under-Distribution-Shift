import { useEffect, useState } from "react";

// Runs `fn()` whenever `deps` change and returns { data, error, loading }.
// Stale responses (after a run switch or unmount) are ignored.  Pass
// `enabled = false` to skip the call (e.g. no run selected yet).
//
// `loading` is true from the very first render in which the inputs changed,
// not only after the effect starts, so callers never see "no data, not
// loading" for one frame and crash on `data.something`.
export default function useApi(fn, deps, enabled = true) {
  const key = JSON.stringify([...deps, enabled]);
  const [state, setState] = useState({ key: null, data: null, error: null });

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    Promise.resolve()
      .then(fn)
      .then((data) => !cancelled && setState({ key, data, error: null }))
      .catch(
        (e) =>
          !cancelled && setState({ key, data: null, error: e?.message || String(e) })
      );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (!enabled) return { data: null, error: null, loading: false };
  if (state.key !== key) return { data: null, error: null, loading: true };
  return { data: state.data, error: state.error, loading: false };
}

//! Windows が報告する現在・最大ワーキングセットを MiB に変換する。
/// OSのカウンタを読み、現在値と起動後の最大値をMiB単位で返す。
#[cfg(windows)]
pub fn working_set() -> (f64, f64) {
    #[repr(C)]
    struct Counters {
        size: u32,
        faults: u32,
        peak: usize,
        current: usize,
        other: [usize; 6],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            stats: *mut Counters,
            size: u32,
        ) -> i32;
    }
    let size = std::mem::size_of::<Counters>() as u32;
    let mut c = Counters {
        size,
        faults: 0,
        peak: 0,
        current: 0,
        other: [0; 6],
    };
    // 現プロセスの疑似ハンドルに対して、構造体サイズの出力領域を渡す。
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, size) };
    assert_ne!(ok, 0);
    (c.current as f64 / 1048576., c.peak as f64 / 1048576.)
}
#[cfg(target_os = "linux")]
/// Linux の /proc/self/status から現在 RSS (VmRSS) とピーク RSS (VmHWM) を MiB で返す。
pub fn working_set() -> (f64, f64) {
    let mut rss = f64::NAN;
    let mut hwm = f64::NAN;
    if let Ok(text) = std::fs::read_to_string("/proc/self/status") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()) {
                    rss = kb / 1024.0;
                }
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()) {
                    hwm = kb / 1024.0;
                }
            }
        }
    }
    (rss, hwm)
}
#[cfg(all(not(windows), not(target_os = "linux")))]
pub fn working_set() -> (f64, f64) {
    (f64::NAN, f64::NAN)
}

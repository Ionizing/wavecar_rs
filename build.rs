#[cfg(target_os = "macos")]
fn main() {
    println!("cargo:rustc-link-search=/usr/local/opt/gcc/lib/gcc/10")
}

#[cfg(not(target_os = "macos"))]
fn main() {}

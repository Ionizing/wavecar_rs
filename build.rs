#[cfg(target_os = "macos")]
fn main() {
    println!("cargo:rustc-link-search=/usr/local/opt/gcc/lib/gcc/10");
    println!("cargo:rustc-link-search=/usr/local/opt/openblas/lib");
}

#[cfg(not(target_os = "macos"))]
fn main() {}

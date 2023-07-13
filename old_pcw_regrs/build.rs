fn main() {
    // pkg_config::Config::new().probe("lapack").unwrap();
    // println!("cargo:rustc-link-lib=/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.9.0");
    println!("cargo:rerun-if-changed=build.rs");
}

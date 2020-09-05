use std::path::PathBuf;

use clap::{Arg, App};
use colored::*;

use wavecar_rs::Wavecar;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("WAVECAR to CHGCAR converter.")
        .version("0.1.0")
        .author("Ionizing <petersmith_9@outlook.com>")
        .about("A tool to extract WAVECAR info and convert it to CHGCAR like files for visualizing.")
        .arg(
            Arg::with_name("list")
                .short('l')
                .long("list")
                .about("List the meta info of specified WAVECAR.")
                .takes_value(false)
        )
        .arg(
            Arg::with_name("input_file")
                .short('i')
                .long("input-file")
                .about("Path of input WAVECAR file.")
                .takes_value(true)
                .default_value("./WAVECAR")
        )
        .get_matches();

    let wave_fpath = PathBuf::from(matches.value_of("input_file").unwrap());
    let wave = Wavecar::from_file(&wave_fpath)?;

    if matches.is_present("list") {
        println!("LISTING META INFO OF CURRENT WAVECAR ...");
        println!("FILE NAME: {}", wave_fpath.to_str().unwrap().bright_green());
        println!("TYPE: {}", wave.get_wavecar_type().to_string().bright_green());
        println!("NSPIN: {}", wave.get_num_spin().to_string().bright_green());
        println!("NKPTS: {}", wave.get_num_kpoints().to_string().bright_green());
        println!("NBANDS: {}", wave.get_num_bands().to_string().bright_green());
        println!("MAX NPLWS: {}", wave.get_num_plane_waves().iter().max().unwrap().to_string().bright_green());
    }
    Ok(())
}
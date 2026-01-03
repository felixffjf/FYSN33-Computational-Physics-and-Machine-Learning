// This program generates Pythia events and writes final state
// particles to a file. To be used for jet
// finding exercise

#include "Pythia8/Pythia.h"
using namespace Pythia8;
int main() {
  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  pythia.readString("Beams:eCM = 13000.");
  pythia.readString("HardQCD:all = on");
  pythia.readString("PhaseSpace:pTHatMin = 20.");
  // Output file
  ofstream ofile("pythia.dat");
  ofile << "# Format: event_id pdg_id px [GeV] py [GeV] pz [GeV] e [GeV]" << endl;
  // If Pythia fails to initialize, exit with error.
  if (!pythia.init()) return 1;
  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < 100000; ++iEvent) {
    if (!pythia.next()) {
      iEvent--;
      continue;
    }
    // Find final particles and write them to file.
    for (int i = 0; i < pythia.event.size(); ++i) {
      Particle& p = pythia.event[i];
      if (p.isFinal() && abs(p.eta()) < 2.5 && p.pT() > 0.5)
        ofile << iEvent << " " << p.id() << " " << p.px() << " " << p.py() << " " 
          << p.pz() << " " << p.e() << endl;
    }
    
  // End of event loop. Statistics. Histogram. Done.
  }
  ofile.close();
  pythia.stat();
  return 0;
}

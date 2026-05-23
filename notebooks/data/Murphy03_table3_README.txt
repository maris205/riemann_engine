J/MNRAS/345/609     Variability of the fine-structure constant (Murphy+, 2003)
================================================================================
Further evidence for a variable fine-structure constant from Keck/HIRES QSO
absorption spectra.
    Murphy M.T., Webb J.K., Flambaum V.V.
   <Mon. Not. R. Astron. Soc., 345, 609-638 (2003)>
   =2003MNRAS.345..609M
================================================================================
ADC_Keywords: Atomic physics ; QSOs
Keywords: atomic data - line: profiles - methods: laboratory -
          techniques: spectroscopic - quasars: absorption lines -
          ultraviolet: general

Abstract:
    We have previously presented evidence for a varying fine-structure
    constant, {alpha}, in two independent samples of Keck/HIRES
    quasi-stellar object (QSO) absorption spectra. Here we present a
    detailed many-multiplet analysis of a third Keck/HIRES sample
    containing 78 absorption systems. We also re-analyse the previous
    samples, providing a total of 128 absorption systems over the redshift
    range 0.2<zabs<3.7.

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
table2.dat    130       46   Atomic data for the MM transitions in our analysis.
table3.dat     59      128   The raw results from the {Chi}^2^ minimization
                             procedure
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1-  6  A6    ---       Ion       Ion (1)
   8- 12  F5.2  ---       A         ? Mass number
  15- 24  F10.5 0.1nm     lambda0   Laboratory wavelength
  26- 27  I2    10-6nm  e_lambda0   ? rms uncertainty on lamba0
  29- 38  F10.4 cm-1      omega0    Laboratory wavenumber
  40- 42  I3   10-4cm-1 e_omega0    ? rms uncertainty on omega0
      44  A1    ---     r_lambda0   [abcde] Reference for lambda0 (2)
  46- 66  A21   ---       Ground    Ground state
  68- 91  A24   ---       Upper     Upper state
      93  A1    ---       ID        Identification letter for the transition,
                                     used in table3.dat
  96- 99  F4.1  eV        IP-       ? Ionization potential for the relevant ion
     100  A1    ---       ---       [,]
 102-105  F4.1  eV        IP+       ? Ionization potential for the ion with a
                                       unit lower charge
 108-112  F5.2  ---       RS        ? Relative strength of the isotopic or
                                       hyperfine components (3)
 113-119  F7.5  ---       f         ? Oscillator strength (4)
 122-126  I5    cm-1      q         ? q coefficient (5)
 128-130  I3    cm-1    e_q         ? rms uncertainty on q
--------------------------------------------------------------------------------
Note (1): The Si II, Al II and Al III wavenumbers have been scaled from 
    their literature values due to the Norlen/Whaling et al. calibration
    difference.
Note (2): References:
    a: Pickering et al. (1998MNRAS.300..131P)
    b: Griesmann & Kling (2000ApJ...536L.113G)
    c: Pickering et al. (2000MNRAS.319..163P)
    d: Pickering et al. (2002A&A...396..715P)
    e: Nave et al. (1991, J. Opt. Soc. Am. B, 8, 2028)
Note (3): The relative strength of the isotopic are form Rosman & Taylor
    1998, J. Phys. Chem. Ref. Data, 27, 1275)
Note (4): Oscillator strength from the DLA data base^4^ of Prochaska et al.
    (2001ApJS..137...21P)
Note (5): q coefficient from Dzuba et al. (1999, Phys. Rev. A, 59, 230;
    1999, Phys. Rev. Lett., 82, 888; 2001, Lecture Notes in Physics,
    Vol. 570, p. 564 and 2002, Phys. Rev. A, 66, 022501)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label         Explanations
--------------------------------------------------------------------------------
       1  A1    ---     Sample        [lhn] Sample (1)
   3- 11  A9    ---     QSO           QSO name (HHMM+DDSS, B1950)
  13- 17  F5.3  ---     zem           Emission redshift
  19- 25  F7.5  ---     zabs          Nominal absorption redshift
      27  A1    ---   n_zabs          [b] Note (2)
  31- 44  A14   ---     Trans         Transition, with letters as defined in
                                       table2.dat (ID)
  46- 51  F6.3  ---     Dalpha/alpha  {Delta}{alpha}/{alpha} ratio (3)
  53- 57  F5.3  ---   e_Dalpha/alpha  rms uncertainty on Dalpha/alpha
      58  A1    ---   n_Dalpha/alpha  [*] *: systems included in the
                                              'high-contrast' sample (4)
--------------------------------------------------------------------------------
Note (1): Samples:
    l: Previous low-z sample
    h: Previous high-z sample
    n: New sample
Note (2): This absorber contributed by Outram et al. (1999MNRAS.310..289O)
Note (3): {alpha} is the fine-structure constant ({alpha}=e^2^/hc)
          {Delta}{alpha}/{alpha}=({alpha}z-{alpha}0)/{alpha}0,
    for {alpha}z and {alpha}0 the values of {alpha} in the absorption
    system(s) and in the laboratory, respectively.
Note (4): An additional random error of 2.09x10^-5^ should be added in
     quadrature to these systems to form the fiducial sample
     (see Section 4.2 of the paper).
--------------------------------------------------------------------------------

History:
    From electronic version of the journal
================================================================================
(End)                       James Marcout, Patricia Vannier [CDS]    13-Feb-2004

"""
Timed sections of the steps of the nonhydrostatic models (`section_timer!`), the
backend synchronized at their start and end so that GPU time counts in its section.
"""
module dmplexTimerOutputsExt

using KernelAbstractions
using TimerOutputs
using dmplex
import dmplex.Nonhydrostatic: section

function section(f, t::TimerOutput, m, name)
    KernelAbstractions.synchronize(m.backend)
    return @timeit t name begin
        r = f()
        KernelAbstractions.synchronize(m.backend)
        r
    end
end

end

"""Offline architecture checks only; embedded Apache-2.0 upstream reference.

No network access, checkpoint, optimizer step, dataset score or report writing.
See speech_denoising/models/tfgridnet/SOURCE.md and LICENSE.upstream.
"""
import base64
import hashlib
import json
import zlib

import torch
from torch import nn

from speech_denoising.models import build_model
from speech_denoising.models.tfgridnet.model import TFGridNetModel
from speech_denoising.metrics import evaluate

REFERENCES = json.loads('{"espnet2/layers/stft.py": {"sha256": "827763e26be6ede1556414d6918763385774bfa39afbc6229267fbe67980d7d6", "compressed": "eNq9WG1v2zYQ/q5fQbgfJrWqWnvYMATwsK1ogA1YCqzevhSBQEuUTUQmNZJKmhX97zu+SHyxnWSvBgKJ5HPHu+Pdw1M6wQ9I3Q+U7RA9DFwo9G5QlDPcl2gzDj0p0a8MxlnmlhUXzT7rjJx+rRt+ANjHShEmuZi0vLGzGzPp4PcD2Y1YtBNGTzR70tyQNrMQIgdG1KoibF/1+J4IWTn19ahoL2dJPm0byzkZym7hQbc9qSlTRHS4IZPoj/Paj9NSooPscHNfwbushnvr4xaDkaw1k7ElB3xD6gG39QHLmyzLmh5Lid6rTuVGtGKs+pm3ow7kia2LiwzB77swFHqiJR2qwXiq6jo3M/onSd+V84jVXacuEHiI1uir5cqv3FFW94Tt1P5iPs4PgLsG4BVnxCP3fJiRVtFy9U2kqOV3gRKphFay2GPGFh7XEO3QBdpy3sPyRozBHoyLA+7pH6SdAZe4lwECTJK0Dda9Ahcg4/04EJEX1RyXIgpMZeIBwuY5L9EuCAei0gTAK52lA1CqgoCxj0n4QWyUDzCA/CAG2fABwL4kbs3h04bNgxg0RRAg02sSAThH7T3jykQAYdaawR5LrJSw2VqibvHJgj/X9rkoYtcFppKg33A/krdCcJF7iWQbqgv0oD1qF8lROeDaSWRzzpOPSuBakEHkGhjsLYgaBUN5ZEy3MCe1/uQT4HOJFgnGn40D+okTaH9IDu0nTqDtiTmkHZxA+WObTJ0nTqCnE3TYafjZ4wofsY6LO+DUhCSgmIcRyMFykGVhmAQ3ZDzpGMHWGnr5rSX9D7HcXP7h9PW1P5zFYvF+c7mZjEHdyBotUmUz5Huxk3EmORPzH7DSmXclsU4YWSCwKpks0RvNOFCIRazCOmTRfukXkyvJdnxU0X6XAh+I1E/ye4lW0bbT2rSrR2Whz/P7VkIcjT+VhEPNXxdh9dmFlh7yAq3X6MvYrsPYK1o3difHfRHg2ZlIBUHRB2dX0fPAaAcsjsM+mwvlxuTAJcmX2r1KELnHA8lfLsvQoWVRPECHqQuG3n2knqGrd5u3+Q0+8KCgn8HfZk90EmOQR1uyx7eUjwLxzqWohGtUs4m+67HSFycQBjBpT7eCS2zWI314C6eM4DpudUOjKU42UGZsV0WwK64IUnsMytUXErW064iA2kWmEZANHe7B7R1kvN0iED7OIp0XU76s1oLgvqYHvCtCIZ9Z7nCeJAipE7JlwN5x+O16rYsOgr8jKiH0QIln9RMaZj42qmKmPXHrlajVbcvapbd+hylyS5t5zgwiPQ+l0WyFYSRPaHAE9Q3wyk5XWUsbFZtmrwB/A5Sp0onOU/sjXED7CevHOEf4Ad+XJ5xY20e8FNwCySUQ42b+j+i/zI5jGITmw8LekFNjvLhOucSV4U+UNfsRswK9GS+Bs11apTW27XqO1fLrLOZPUOprM7fHbJB5UaLnzwODinOSt5Tc1VjWOuNzu1ZUOnvyIJOKf1x0MyUeUf0p/GyjfUl5MazJiO0uEqqOTfVs/CQjTht+RBpnZVMndKTzLeyp2dzNOTqPx6vCkL93+qj2dRSSQo6uOH0Rn2eoicncd8KRdmBssDro4l+9Srabb3t9c5nnC7SC8IJkFsfAgY5dsNnX0tvcKngZbFim3XqJBB+ZvkbqA2/JeqEEcGJCm/r3Ai3PEJw/kUp/HJK27mjf13n0yZgbe6fTKJE+mdfV64docvIwZknXH096DMh3ifaLmJzpEs0XftLzRR/w1/9t72g/jAnSPeRTWkarJN/aItlAXUwNXGR1CDjdN1rAI43jHb71WDR1qU/SF/aIU9/l/3dh+S4upP/pwm8xyADNzn2gZhTLvI83Bmf6gEnnySbgb9z7DzaqHqaPZ75bpg8PcIaaG+o4fcq/0Dw8tSl4apPxQHPwWF/xbzUP+udMNVkL3PQRvktOcbg5Kjsdy8edxjr5b05xREr6iByDZH8CSxL63g=="}, "espnet2/enh/encoder/stft_encoder.py": {"sha256": "bb76ea25b4cb02b26bc668887f540d5952c714c746ebf6cfc9299723b807ff45", "compressed": "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"}, "espnet2/enh/decoder/stft_decoder.py": {"sha256": "4187de232932da18804392b08bf8de3dabdcbf7bca39c6c9e1a46bed10ea8e79", "compressed": "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"}, "espnet2/torch_utils/get_layer_from_string.py": {"sha256": "f8821d9566f326f75e69339eca18fa3b84d81b84915502dddbfe64b30dc0884e", "compressed": "eNrVVMFq3DAQvfsrBvcQG4yaXhdcCGEPhTSHQk/pYpT1eK0iS6kkZzeE/HtHkq21k4b2Wp8szcx7b96MLYYHbRy0ouukuM8yEc9Om32fZVmLHRzQNZI/oSlko/iAFVCm4eapDllMqXKTAT15nn9DNxoFIR30/U/cO+i5aiUdO6OHuRSQHVi8mUGygLH196KDSFXXOcoxr8AEXJuS2fbmO4sVV+ZgI79/Yh0U1hmhDuUGrrlFEMqissKJR4QQ77SZRAqVNBVBFLsgyouSnSHn8KDbUSJh3noI3c2RjzEAxx4NkkSwyElmIFl7kDCPwvWz1sCaz53lU1vRyWVnXm4zm1lEYBKT7CU21yNZ9WtE67CdGgzwxdK4MpvHFV+4lE2IxzFbqOHuFPBO3p5WmFRe7kLFwB31t057g5KmyKQ+0vaUUNdwmg8RyOegKgJeiF+eG95LbbEJIfSaph1lfh9XsSKVnDegIm2J912Nu1RZpjfDBW3MrXZfhgeJAyqycmuMNmua/Ca4GwYZxvj8Ako74hpV67meX9gPBddeqHUwSd3Qdc5ID53XeEvptLzFtFtltfZhVVO+ko9y7edn+PTf2Pl1lE5QRnTKLr7Qv1mcr4GupPyj2/9g7ls/LZ4N/ACPXIr2nQ+y9v9J7s7gVVRxd7lbmLH4Pc6V2W95kbBw"}, "espnet2/enh/separator/tfgridnet_separator.py": {"sha256": "e08616b3e1964c7b0019320fff47110a4708fff44752173faff95aefed50e1f6", "compressed": "eNrtG2tv20byu37FwkUR0qEYS3aK1oAO1yRWEzh2zrHOvZ4gELS4sljzoXKXtpzD/feb2V0ud/mQnTbt4Q412ork7rwfOzNk43STF5ykIV8PVkWekmWeJHTJ4zxjJJaLH4qIFjR6Ey+53MMfNnF2Uy3jc4+8jxn898MGIcPEI7Nyk9DBQO3hebFcWzd+lpGQkSxrPvVXZbaUWHDDVJGsYNTuOIu5veJvwiJMKadFtedv1YOB3EnZJqN87NNs7Ud0mYNUPuMrHqibCu5yNp29kY/agDQzANWNCXiS9QAm4QMtmL/MU1DMNih5nGgNZ/Q+qBaS+Ja2oRlF8UBWP7xmgb6rEHx/zS6rZzawUI+k5t9QHgg2AtwSMF4YZtSLg8FgmYSMkdn0hyKOzil3TPTu8YDA397e3ofVKokzWu8biJWPdAXuki2p3DgfLcg//eGFT34MsxuPXPrkdV5kNEnk9TqPPfKTT95T6pFX/hAuT+PUI2EW4fqPIQ+z8Jp6kupsOlTEjsm7jNMb4AmFmJZJMpQw5fXwFV6cgR0SXFuBni43lC7XREkB3rUnEcbghcU/4juyKegG1MHl7fF4PBr5o/HR4aFHxgfjsS9lGf9espyFt8jqLE7pcFrQX0pQ4AN5k6chcCgkYeSHgoZg6ht8hDKdQYyURZgIbCBgeAuuaEq4Q7qD7/yDw+9ejivpBI7zD7OTS2m17xnJS47mjRALX9Parh7cxoykyBW5z4tbRq4p4+R+TWFnEQqg+5ivyV1YxCFACJRZXqRhEn+CxTTe8rKggHlTcqEdHhbggB6h/o3vSeBqU74ibB1uKJlfh3y59ggLMVKY1GIaL4t8s84zyhYeecjLmg6JObl+IFF8F0eoXIFVpgvGI0fh94gzAi24rk9+ykuJswRpolxIDcSo0DbeSC4Ji28gOzGfvINExAhmAbDDDQjGc4RjudRFlnNSMiAtsLJlmNBhnEmlcJLkEGNVsmOEleCfkPAu3w0v33xUFvm+uGHSINJZQV1BFKfHZJOES7rOE0g1nqJDI70xC1ixZMckK9Nr8AnQIBgTVc2A0SVlL5j0FuYbIKsV+CGmNVBUFuX3IOYnWm/AbBFRtUPe1IsSwgaHU4KS5TrPGQX/4PcUNPJsHaYpKOSZh5dZhpcElHsO9jN5icGsFv+GmQEnQGJAOHmWPJBVvKXRMCyK8AGSWA4Zv3gAbW4wq9HINdHKHGzi1bmLXCf58tZQSMJ4GqzjKKJZUMJpY4HJ50Q8x/h4fzk7q0FDzrMgC9Y0jCwguBe7GU1WQ9hEMzS9DRZuNkW+DX65lXaWtzGczxQcOZUQiGyFZ9swoXcQg7f0QQTRXZiU4KQ0Y3nBNFqaXktccEEjEQkCEzNp46ZbEBGcAlQrTC+8vsxW4GQIgwTwoMzuRm8ssDWArfPNk2FCcPg7kaGOjWsdCRhD4MxV1rkHJzeOGJl1PI0M/zDol2EmoMLsQYZ47QEmEUwv5FlBk1K43TO8qO1GNyAKg+SR4GWc5DLPVglFolDnuAYCqsF1GSdwDFVn+DGGPzBfVLKodaLWZWgIzNyvTlMZ8BFdkSDA6iYInDr2wGG8dhrwGgE/GXt2PE9G42+9RgRPvjnyGnE72cOI2vOaATgZea3gmXzj9YfIZPSdwYIRBpMjb4ebT16ODDDlr5Ojb72Ge5pYpOeZHNZmnuxt0MSGQGDOycg/oMOX3i7DTaaQ1mm9paCrQGWbyVCRcut8DC5GC8f1tcFcy2K+tAqZKPM0F6VCxbK8bG4QRhDr4qqWlDGKVSOamHxNxmQyIQem5aGAkHC44cUL2PGcjGzshmiw07gb2PvgcIN1o7J1rNgTJLzqR3qYV7mV/PE6Nd3xTCNuqBEyiGJBVeXOE6i6tRwcXAdz04Qc6me3zCObUGbDCXHUFo8cup6+E5rzyKjBDqYzVG7mX4o6jcdh0lBK5r+GTePIGZP9ynpe5dieSXyifl2vieGHIi8355B6sDzRoOjI8K+x3W0YTJ5jkkEoHMuEYmvmzBe1GJjUAkywBVSx1Kncz/DsBjYfwpVmkS0l/qms/Ap3tVeNaPZ6F29Z/9q6Z025ePdiKyv14RCJyUhS3fvsTNVOXj1QdTKqL3sElSZtL7oD+64VFdoR0dtmYEy2gWIL3E67i8pA++jHXV5XHzrgE/dhEe08c45V9TwTFYaxmMCDvkWkJNv5Yz0emOPMYAG8Y92n0ioZ/kUODebor3MTGRT2Fm5zIrGonRYO0amUwq91ZVfQWhQIcgOje0xEb4GtSQkpaVhlxrCM4lxVVKJ/6LTgmdqnC1Ose86rPoXMX3nk3CNnC5sNVFqLDckcLN3wNULaMLUuiYPCV6UzAOai3oigw/TIsiygUeNQG5cZ9gWylqpaNkM5Hyl0QFlDPzRbY88WEUcY4u8ZELTZXDQyRfU3d0DSGWRQ3/cXJAForFQT7ITsA9D8w+I+z3JL1Uz2arNKhf4TNAe03S+vrO6QvadwYqLqsMWEkiFHINli+aY7mvWZcoaJNLEvWtr5qLZvvGoc+xMystWsjn2Q3jFwuLhz3OHhitIcjOEJwReEfGU4owaBRjiAhjgAiLo7FrAedgoT2R9DAqF0g/ezogSqChesmUKYlMmLGjXs/nh2aZfSNQci+ACsqjgq6sLS7vxAcz4Dzj0yXdiQBwAqLnxeJUHJcwV2JiCbYFreZcgdR2KCuginl+oG+i44mYUSRhrZeL+FLgvUaCIQRQm2ZswzCjHJnDBXj9CYyyULmg4UnIqMdXDHcX1yW4VkIygt9PIYn8fx4nEiFqA8ZRpAMpr3x72gUt67mN47c60boyT28DxqKcqoUCpEzcGoshLUaOJifuwR+OdgoSym7kcL1+2XxzG4Q/FbfLgdjlfxPRz1aQqO1LGzQ/AyDdjmVmBwvTohuD3agzPbDJ4CGv0COkkxiHs8kOZaHcD2QvgNXDQdR7LkGhKprKYYF1qwTltHsftXKICg9+EPun5QyARmwxMVQsP2CgHjwPoyhTY5jzQSoUPZ3nJ11MuRWwCcuM1BmDqWJ12vDtAcTjsparTOgYmaDK1llZzRUh09SZX5DQA9MrfqYV2AK85lbw8kY05T6BZFhYXDm7a+YBNUmoWx5enjgVbF3ayym5V1q5pWxXNVzpg7W818fx//aD+uu/HdLbVhdc0SGF2JCXEi5WuUx3EGZ0GAYQJ732OCPDdD5uiN0+qpuhAUWSZLbJzwNR2q1lBLZXguiigKVnHBuDgz4UkcxQVVLiqe9fW8kjrO38Oio8YfNZy7QV4W/LWA0gGqAZC0f28DCaRp8Rs1Bwj+i5oD6n+05k6AFL5E9Zc0bkwDrBAB/CP/ACojFXQ1Lszy9laoQcWA25gBN0dAVRB8rULTngQ1ywW5p6vNh1o5SEWuanfxe6LnxTIguAi+jvaAWBy36+Id85D2XEQr+ATnK91Ftn4h6dSZxHV6Nnf56eup45yYp3p7eKK17w36O+/P0dDpnxp6RENXX1xDVRC8eKGi4HfWWAe936TBndrTmoMmJYA88PNea1K4S2X96upW0pMV9IhyHlfJjhFmpWB9yLeXb6sS4Ja1F9fV4ro1UpdJUtmtPf2S9dbWtYZKZlFHnjJh2h6LnuG16I8u7BFOjmO0zlVzXiDX8yQKZvLnApjeqg6y2jSzzhxH7IZK1tCRCweNoRUXTh9TSc/NvRrtRRvtxRdAuwW0U1GWb2X5jUgFcmjgQBfqbma2bl8RUQeRj+fnjdof1WE8qku1RvGnugndura0bsI2qxS10pwqYLPO45syLxlUqqLZewUqmAnsF659pM9fqefdJKe+fFvaQ9oY3oDSDLWK6K3KEsfQvvW+wuBgv4Ibjro56RVX48BmucLTgcMjga1/eOboKxvN21/Lw9t+AVp1cxfxfkvUtGX7LqdNYr/7ecxa/vEkx6uvn1f+3QSzY4IW3TGhEZlkdDHeKO+7Y2O66IRtOahc8aFdS0tOMaAPRe3eGx8XkkLLO6fyeTfZ3viQO75MfEx3xIfmpBa50zenO+JDwtnxIZ85+spG8/bX8vC2X4BWd9RFvN8SNW0dH8qidnx0ABg+MpZu8vQYaaN7Uoy0v67pxCX951gdtOL3oj0l1ttrCmGS4MGFP6fy5wrHbuCF8t/HRrYdjZhAWb1pxW3zzs6rmuD2a61CdtqH7PRXILvqQ3b1BGRWbVHP3JUScXJ0oICedbBw2gI6fRzoqgV01Q9k8XfRDK7G6ioRzuUwHhbiM5zJuEYpjtoG76e9COXqZyFEua66w/9ZfWLp7ejR8utJBLPrR4nqs6jXlflFPSQd2J/2QO2odQ/XaZk4YOSWErCIrLqF/f0D/6VF1UhCBtKpz/IVT8OtUz2UNh33wV41OanhrtpSNjRTUCGio3W4Q8+7zdLyzlqPV/WbwFbeuVKp1sgZHrYGusdSeQsH1YKWsaci6fa8nKlZPcB8XL/nkBgeA28f8VbGQFRmx7XfxXPzGN5F1c46uhduvs6qEQyMhku/U3lupPLGxB226Sl+59Cze5qvptSyb9Qf5smPKz095n7aN2Pif2AI1KdK81EHPvtlq1DwTZimIezX/7OD9V7c2a+Ruj7P1eIqyUN+CDHY+HSI8i+ACuXy8UNdqRbJYWP5Ey3yah2pNhgBvWEHv2FP6dDjFXTGmYimCTmyz9RAuEQgS69m/yy/iQy5ikSI2ZoNmjBqoyrCmFFyhZ/YnhQF6GPvZLuhSw6NLc/JOryj5Kj+sBYKyWtwvJuck3/9e89foTNxR/Jp6CotJWMpDTMHM1nFT/fb9pE384z63H5p/0vB7SDc+ndQ5NlYy+w6DhmN1JeODSrPtfpbsdkgvQ3WIiM7zhYadxBDzASAHz0MkI75vPasZsQJFDtj7vX0D4i65lcVNabGlxWPheccX0N3PB4t/gzZzwhZOyLJ4f9CTI7+v2LyP3xtD9Y="}}')


def reference_class():
    # Execute the pinned originals after removing only optional ESPnet imports.
    # The reference uses its own original STFT encoder/decoder, not our frontend.
    class LegacyComplex:
        pass

    class InversibleInterface:
        pass

    def make_pad_mask(lengths, tensor, length_dim):
        if length_dim != 1:
            raise AssertionError("Reference fixture only uses frame axis one")
        indices = torch.arange(tensor.shape[1], device=tensor.device)[None, :]
        mask = indices >= lengths.to(tensor.device)[:, None]
        return mask.view(*mask.shape, *([1] * (tensor.ndim - 2))).expand_as(tensor)

    namespace = dict(__name__="_pinned_tfgridnet_reference", AbsSeparator=nn.Module,
                     AbsEncoder=nn.Module, AbsDecoder=nn.Module,
                     ComplexTensor=LegacyComplex, InversibleInterface=InversibleInterface,
                     typechecked=lambda function: function, make_pad_mask=make_pad_mask,
                     to_complex=lambda tensor: tensor if torch.is_complex(tensor)
                     else torch.complex(tensor[..., 0], tensor[..., 1]),
                     is_torch_complex_tensor=torch.is_complex,
                     new_complex_like=lambda reference, parts: torch.complex(*parts))
    for name, record in REFERENCES.items():
        raw = zlib.decompress(base64.b64decode(record["compressed"]))
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        lines = [line for line in raw.decode().splitlines()
                 if not line.startswith(("from espnet2.", "from torch_complex",
                                         "from typeguard", "import torch_complex"))]
        exec(compile("\n".join(lines), name, "exec"), namespace)
    return namespace["TFGridNet"]


def assert_equivalence(model, Reference, audio):
    reference = Reference(input_dim=0, n_srcs=1, n_imics=1, window="hann",
                          use_builtin_complex=True, **model.config).to(
                              device=audio.device, dtype=audio.dtype).eval()
    state = model.network.state_dict()
    assert set(state) == set(reference.state_dict())
    reference.load_state_dict(state, strict=True)
    lengths = torch.full((audio.shape[0],), audio.shape[1], device=audio.device,
                         dtype=torch.long)
    with torch.no_grad():
        ours_spec, ours_lengths = model.network.enc(audio[..., None], lengths)
        ref_spec, ref_lengths = reference.enc(audio[..., None], lengths)
        torch.testing.assert_close(ours_spec, ref_spec, rtol=0, atol=0)
        assert torch.equal(ours_lengths, ref_lengths)
        ours_roundtrip = model.network.dec(ours_spec.squeeze(2), lengths)[0]
        ref_roundtrip = reference.dec(ref_spec.squeeze(2), lengths)[0]
        torch.testing.assert_close(ours_roundtrip, ref_roundtrip, rtol=0, atol=0)
        torch.testing.assert_close(ours_roundtrip, audio, rtol=1e-5, atol=1e-6)
        reference_output = reference(audio, lengths)[0][0]
        actual = model(audio)
    torch.testing.assert_close(actual, reference_output, rtol=1e-5, atol=1e-6)
    assert actual.shape == audio.shape and actual.dtype == audio.dtype
    assert actual.device == audio.device and torch.isfinite(actual).all()


class Fixture:
    protocol_complete = False
    split = "synthetic_fixture"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        clean = torch.sin(torch.arange(1025, dtype=torch.float32) * 0.1) * 0.1
        return dict(utterance_id="synthetic_tfgridnet", sample_rate=16000,
                    noisy_audio=clean + torch.cos(torch.arange(1025) * 0.17) * 0.01,
                    clean_audio=clean)


def main():
    torch.manual_seed(27)
    torch.set_num_threads(1)
    Reference = reference_class()
    default = build_model("tfgridnet").eval()
    assert isinstance(default, TFGridNetModel)
    assert default.minimum_samples == 768
    assert_equivalence(default, Reference, torch.randn(2, 1025) * 0.1)
    assert_equivalence(default, Reference, torch.randn(1, 16001) * 0.1)
    with torch.no_grad():
        batch = torch.randn(2, 1025) * 0.1
        torch.testing.assert_close(default(batch), torch.cat([default(x[None]) for x in batch]),
                                   rtol=2e-4, atol=2e-6)
    total = sum(p.numel() for p in default.parameters())
    trainable = sum(p.numel() for p in default.parameters() if p.requires_grad)
    entries = len(default.network.state_dict())
    del default

    small = TFGridNetModel(n_layers=1, lstm_hidden_units=16, emb_dim=8,
                          attn_n_head=2, attn_approx_qk_dim=64).eval()
    for length in (1, 2, 255, 256, 257, 513, 767, 768, 769, 1025):
        audio = torch.randn(2, length) * 0.1
        with torch.no_grad():
            actual = small(audio)
            silence = small(torch.zeros_like(audio))
        assert actual.shape == audio.shape and torch.isfinite(actual).all()
        assert torch.equal(silence, torch.zeros_like(silence))
    with torch.no_grad():
        assert torch.isfinite(small(torch.full((1, 1025), 0.1))).all()
    for invalid in (torch.empty(0, 1025), torch.empty(1, 0), torch.randn(1025),
                    torch.randn(1, 1, 1025), torch.ones(1, 1025, dtype=torch.int64),
                    torch.full((1, 1025), float("nan")),
                    torch.full((1, 1025), float("inf")), torch.randn(1, 1025).double()):
        try:
            small(invalid)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid audio was accepted")
    small.double()
    assert_equivalence(small, Reference, torch.randn(1, 1025).double() * 0.1)
    small.float().train()
    audio = (torch.randn(1, 1025) * 0.1).requires_grad_()
    small(audio).square().mean().backward()
    assert audio.grad is not None and torch.isfinite(audio.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
    small.zero_grad(set_to_none=True)
    report = evaluate(small, Fixture(), metrics=("si_snr", "si_snr_improvement"),
                      warmup=0, repeats=1)
    assert small.training, "Shared evaluator must restore model mode"
    assert report["scope"] == "subset_or_fixture"
    assert not report["failed_utterances"]
    # Discard synthetic random-weight scores; do not export them as benchmark results.
    if torch.cuda.is_available():
        small.cuda().eval()
        assert_equivalence(small, Reference, torch.randn(1, 1025, device="cuda") * 0.1)
        print("PASS: optional CUDA reference equivalence")
    print("PASS: TF-GridNet upstream equivalence, STFT, waveform alignment and gradients")
    print("PASS: shared evaluate(model, dataset) interface; synthetic fixture only")
    print("Parameters (all / trainable):", total, trainable)
    print("Matched network state entries:", entries)
    print("Offline original TF-GridNet; local single-output 16 kHz profile")
    print("Random weights only; no checkpoint, training or dataset quality evaluation.")


if __name__ == "__main__":
    main()

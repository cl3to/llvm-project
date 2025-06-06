//===- BottomUpVec.cpp - A bottom-up vectorizer pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm {

#ifndef NDEBUG
static cl::opt<bool>
    AlwaysVerify("sbvec-always-verify", cl::init(false), cl::Hidden,
                 cl::desc("Helps find bugs by verifying the IR whenever we "
                          "emit new instructions (*very* expensive)."));
#endif // NDEBUG

static constexpr const unsigned long StopAtDisabled =
    std::numeric_limits<unsigned long>::max();
static cl::opt<unsigned long>
    StopAt("sbvec-stop-at", cl::init(StopAtDisabled), cl::Hidden,
           cl::desc("Vectorize if the invocation count is < than this. 0 "
                    "disables vectorization."));

static constexpr const unsigned long StopBundleDisabled =
    std::numeric_limits<unsigned long>::max();
static cl::opt<unsigned long>
    StopBundle("sbvec-stop-bndl", cl::init(StopBundleDisabled), cl::Hidden,
               cl::desc("Vectorize up to this many bundles."));

namespace sandboxir {

static SmallVector<Value *, 4> getOperand(ArrayRef<Value *> Bndl,
                                          unsigned OpIdx) {
  SmallVector<Value *, 4> Operands;
  for (Value *BndlV : Bndl) {
    auto *BndlI = cast<Instruction>(BndlV);
    Operands.push_back(BndlI->getOperand(OpIdx));
  }
  return Operands;
}

/// \Returns the BB iterator after the lowest instruction in \p Vals, or the top
/// of BB if no instruction found in \p Vals.
static BasicBlock::iterator getInsertPointAfterInstrs(ArrayRef<Value *> Vals,
                                                      BasicBlock *BB) {
  auto *BotI = VecUtils::getLastPHIOrSelf(VecUtils::getLowest(Vals, BB));
  if (BotI == nullptr)
    // We are using BB->begin() (or after PHIs) as the fallback insert point.
    return BB->empty()
               ? BB->begin()
               : std::next(
                     VecUtils::getLastPHIOrSelf(&*BB->begin())->getIterator());
  return std::next(BotI->getIterator());
}

Value *BottomUpVec::createVectorInstr(ArrayRef<Value *> Bndl,
                                      ArrayRef<Value *> Operands) {
  auto CreateVectorInstr = [](ArrayRef<Value *> Bndl,
                              ArrayRef<Value *> Operands) -> Value * {
    assert(all_of(Bndl, [](auto *V) { return isa<Instruction>(V); }) &&
           "Expect Instructions!");
    auto &Ctx = Bndl[0]->getContext();

    Type *ScalarTy = VecUtils::getElementType(Utils::getExpectedType(Bndl[0]));
    auto *VecTy = VecUtils::getWideType(ScalarTy, VecUtils::getNumLanes(Bndl));

    BasicBlock::iterator WhereIt = getInsertPointAfterInstrs(
        Bndl, cast<Instruction>(Bndl[0])->getParent());

    auto Opcode = cast<Instruction>(Bndl[0])->getOpcode();
    switch (Opcode) {
    case Instruction::Opcode::ZExt:
    case Instruction::Opcode::SExt:
    case Instruction::Opcode::FPToUI:
    case Instruction::Opcode::FPToSI:
    case Instruction::Opcode::FPExt:
    case Instruction::Opcode::PtrToInt:
    case Instruction::Opcode::IntToPtr:
    case Instruction::Opcode::SIToFP:
    case Instruction::Opcode::UIToFP:
    case Instruction::Opcode::Trunc:
    case Instruction::Opcode::FPTrunc:
    case Instruction::Opcode::BitCast: {
      assert(Operands.size() == 1u && "Casts are unary!");
      return CastInst::create(VecTy, Opcode, Operands[0], WhereIt, Ctx,
                              "VCast");
    }
    case Instruction::Opcode::FCmp:
    case Instruction::Opcode::ICmp: {
      auto Pred = cast<CmpInst>(Bndl[0])->getPredicate();
      assert(all_of(drop_begin(Bndl),
                    [Pred](auto *SBV) {
                      return cast<CmpInst>(SBV)->getPredicate() == Pred;
                    }) &&
             "Expected same predicate across bundle.");
      return CmpInst::create(Pred, Operands[0], Operands[1], WhereIt, Ctx,
                             "VCmp");
    }
    case Instruction::Opcode::Select: {
      return SelectInst::create(Operands[0], Operands[1], Operands[2], WhereIt,
                                Ctx, "Vec");
    }
    case Instruction::Opcode::FNeg: {
      auto *UOp0 = cast<UnaryOperator>(Bndl[0]);
      auto OpC = UOp0->getOpcode();
      return UnaryOperator::createWithCopiedFlags(OpC, Operands[0], UOp0,
                                                  WhereIt, Ctx, "Vec");
    }
    case Instruction::Opcode::Add:
    case Instruction::Opcode::FAdd:
    case Instruction::Opcode::Sub:
    case Instruction::Opcode::FSub:
    case Instruction::Opcode::Mul:
    case Instruction::Opcode::FMul:
    case Instruction::Opcode::UDiv:
    case Instruction::Opcode::SDiv:
    case Instruction::Opcode::FDiv:
    case Instruction::Opcode::URem:
    case Instruction::Opcode::SRem:
    case Instruction::Opcode::FRem:
    case Instruction::Opcode::Shl:
    case Instruction::Opcode::LShr:
    case Instruction::Opcode::AShr:
    case Instruction::Opcode::And:
    case Instruction::Opcode::Or:
    case Instruction::Opcode::Xor: {
      auto *BinOp0 = cast<BinaryOperator>(Bndl[0]);
      auto *LHS = Operands[0];
      auto *RHS = Operands[1];
      return BinaryOperator::createWithCopiedFlags(
          BinOp0->getOpcode(), LHS, RHS, BinOp0, WhereIt, Ctx, "Vec");
    }
    case Instruction::Opcode::Load: {
      auto *Ld0 = cast<LoadInst>(Bndl[0]);
      Value *Ptr = Ld0->getPointerOperand();
      return LoadInst::create(VecTy, Ptr, Ld0->getAlign(), WhereIt, Ctx,
                              "VecL");
    }
    case Instruction::Opcode::Store: {
      auto Align = cast<StoreInst>(Bndl[0])->getAlign();
      Value *Val = Operands[0];
      Value *Ptr = Operands[1];
      return StoreInst::create(Val, Ptr, Align, WhereIt, Ctx);
    }
    case Instruction::Opcode::Br:
    case Instruction::Opcode::Ret:
    case Instruction::Opcode::PHI:
    case Instruction::Opcode::AddrSpaceCast:
    case Instruction::Opcode::Call:
    case Instruction::Opcode::GetElementPtr:
      llvm_unreachable("Unimplemented");
      break;
    default:
      llvm_unreachable("Unimplemented");
      break;
    }
    llvm_unreachable("Missing switch case!");
    // TODO: Propagate debug info.
  };

  auto *NewI = CreateVectorInstr(Bndl, Operands);
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "New instr: " << *NewI << "\n");
  return NewI;
}

void BottomUpVec::tryEraseDeadInstrs() {
  DenseMap<BasicBlock *, SmallVector<Instruction *>> SortedDeadInstrCandidates;
  // The dead instrs could span BBs, so we need to collect and sort them per BB.
  for (auto *DeadI : DeadInstrCandidates)
    SortedDeadInstrCandidates[DeadI->getParent()].push_back(DeadI);
  for (auto &Pair : SortedDeadInstrCandidates)
    sort(Pair.second,
         [](Instruction *I1, Instruction *I2) { return I1->comesBefore(I2); });
  for (const auto &Pair : SortedDeadInstrCandidates) {
    for (Instruction *I : reverse(Pair.second)) {
      if (I->hasNUses(0)) {
        // Erase the dead instructions bottom-to-top.
        LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Erase dead: " << *I << "\n");
        I->eraseFromParent();
      }
    }
  }
  DeadInstrCandidates.clear();
}

Value *BottomUpVec::createShuffle(Value *VecOp, const ShuffleMask &Mask,
                                  BasicBlock *UserBB) {
  BasicBlock::iterator WhereIt = getInsertPointAfterInstrs({VecOp}, UserBB);
  return ShuffleVectorInst::create(VecOp, VecOp, Mask, WhereIt,
                                   VecOp->getContext(), "VShuf");
}

Value *BottomUpVec::createPack(ArrayRef<Value *> ToPack, BasicBlock *UserBB) {
  BasicBlock::iterator WhereIt = getInsertPointAfterInstrs(ToPack, UserBB);

  Type *ScalarTy = VecUtils::getCommonScalarType(ToPack);
  unsigned Lanes = VecUtils::getNumLanes(ToPack);
  Type *VecTy = VecUtils::getWideType(ScalarTy, Lanes);

  // Create a series of pack instructions.
  Value *LastInsert = PoisonValue::get(VecTy);

  Context &Ctx = ToPack[0]->getContext();

  unsigned InsertIdx = 0;
  for (Value *Elm : ToPack) {
    // An element can be either scalar or vector. We need to generate different
    // IR for each case.
    if (Elm->getType()->isVectorTy()) {
      unsigned NumElms =
          cast<FixedVectorType>(Elm->getType())->getNumElements();
      for (auto ExtrLane : seq<int>(0, NumElms)) {
        // We generate extract-insert pairs, for each lane in `Elm`.
        Constant *ExtrLaneC =
            ConstantInt::getSigned(Type::getInt32Ty(Ctx), ExtrLane);
        // This may return a Constant if Elm is a Constant.
        auto *ExtrI =
            ExtractElementInst::create(Elm, ExtrLaneC, WhereIt, Ctx, "VPack");
        if (!isa<Constant>(ExtrI))
          WhereIt = std::next(cast<Instruction>(ExtrI)->getIterator());
        Constant *InsertLaneC =
            ConstantInt::getSigned(Type::getInt32Ty(Ctx), InsertIdx++);
        // This may also return a Constant if ExtrI is a Constant.
        auto *InsertI = InsertElementInst::create(
            LastInsert, ExtrI, InsertLaneC, WhereIt, Ctx, "VPack");
        LastInsert = InsertI;
        if (!isa<Constant>(InsertI))
          WhereIt = std::next(cast<Instruction>(LastInsert)->getIterator());
      }
    } else {
      Constant *InsertLaneC =
          ConstantInt::getSigned(Type::getInt32Ty(Ctx), InsertIdx++);
      // This may be folded into a Constant if LastInsert is a Constant. In
      // that case we only collect the last constant.
      LastInsert = InsertElementInst::create(LastInsert, Elm, InsertLaneC,
                                             WhereIt, Ctx, "Pack");
      if (auto *NewI = dyn_cast<Instruction>(LastInsert))
        WhereIt = std::next(NewI->getIterator());
    }
  }
  return LastInsert;
}

void BottomUpVec::collectPotentiallyDeadInstrs(ArrayRef<Value *> Bndl) {
  for (Value *V : Bndl)
    DeadInstrCandidates.insert(cast<Instruction>(V));
  // Also collect the GEPs of vectorized loads and stores.
  auto Opcode = cast<Instruction>(Bndl[0])->getOpcode();
  switch (Opcode) {
  case Instruction::Opcode::Load: {
    for (Value *V : drop_begin(Bndl))
      if (auto *Ptr =
              dyn_cast<Instruction>(cast<LoadInst>(V)->getPointerOperand()))
        DeadInstrCandidates.insert(Ptr);
    break;
  }
  case Instruction::Opcode::Store: {
    for (Value *V : drop_begin(Bndl))
      if (auto *Ptr =
              dyn_cast<Instruction>(cast<StoreInst>(V)->getPointerOperand()))
        DeadInstrCandidates.insert(Ptr);
    break;
  }
  default:
    break;
  }
}

Action *BottomUpVec::vectorizeRec(ArrayRef<Value *> Bndl,
                                  ArrayRef<Value *> UserBndl, unsigned Depth) {
  bool StopForDebug =
      DebugBndlCnt++ >= StopBundle && StopBundle != StopBundleDisabled;
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "canVectorize() Bundle:\n";
             VecUtils::dump(Bndl));
  const auto &LegalityRes = StopForDebug ? Legality->getForcedPackForDebugging()
                                         : Legality->canVectorize(Bndl);
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Legality: " << LegalityRes << "\n");
  auto ActionPtr =
      std::make_unique<Action>(&LegalityRes, Bndl, UserBndl, Depth);
  SmallVector<Action *> Operands;
  switch (LegalityRes.getSubclassID()) {
  case LegalityResultID::Widen: {
    auto *I = cast<Instruction>(Bndl[0]);
    switch (I->getOpcode()) {
    case Instruction::Opcode::Load:
      break;
    case Instruction::Opcode::Store: {
      // Don't recurse towards the pointer operand.
      Action *OpA = vectorizeRec(getOperand(Bndl, 0), Bndl, Depth + 1);
      Operands.push_back(OpA);
      break;
    }
    default:
      // Visit all operands.
      for (auto OpIdx : seq<unsigned>(I->getNumOperands())) {
        Action *OpA = vectorizeRec(getOperand(Bndl, OpIdx), Bndl, Depth + 1);
        Operands.push_back(OpA);
      }
      break;
    }
    // Update the maps to mark Bndl as "vectorized".
    IMaps->registerVector(Bndl, ActionPtr.get());
    break;
  }
  case LegalityResultID::DiamondReuse:
  case LegalityResultID::DiamondReuseWithShuffle:
  case LegalityResultID::DiamondReuseMultiInput:
  case LegalityResultID::Pack:
    break;
  }
  // Create actions in post-order.
  ActionPtr->Operands = std::move(Operands);
  auto *Action = ActionPtr.get();
  Actions.push_back(std::move(ActionPtr));
  return Action;
}

#ifndef NDEBUG
void BottomUpVec::ActionsVector::print(raw_ostream &OS) const {
  for (auto [Idx, Action] : enumerate(Actions)) {
    Action->print(OS);
    OS << "\n";
  }
}
void BottomUpVec::ActionsVector::dump() const { print(dbgs()); }
#endif // NDEBUG

Value *BottomUpVec::emitVectors() {
  Value *NewVec = nullptr;
  for (const auto &ActionPtr : Actions) {
    ArrayRef<Value *> Bndl = ActionPtr->Bndl;
    ArrayRef<Value *> UserBndl = ActionPtr->UserBndl;
    const LegalityResult &LegalityRes = *ActionPtr->LegalityRes;
    unsigned Depth = ActionPtr->Depth;
    auto *UserBB = !UserBndl.empty()
                       ? cast<Instruction>(UserBndl.front())->getParent()
                       : cast<Instruction>(Bndl[0])->getParent();

    switch (LegalityRes.getSubclassID()) {
    case LegalityResultID::Widen: {
      auto *I = cast<Instruction>(Bndl[0]);
      SmallVector<Value *, 2> VecOperands;
      switch (I->getOpcode()) {
      case Instruction::Opcode::Load:
        VecOperands.push_back(cast<LoadInst>(I)->getPointerOperand());
        break;
      case Instruction::Opcode::Store: {
        VecOperands.push_back(ActionPtr->Operands[0]->Vec);
        VecOperands.push_back(cast<StoreInst>(I)->getPointerOperand());
        break;
      }
      default:
        // Visit all operands.
        for (Action *OpA : ActionPtr->Operands) {
          auto *VecOp = OpA->Vec;
          VecOperands.push_back(VecOp);
        }
        break;
      }
      NewVec = createVectorInstr(ActionPtr->Bndl, VecOperands);
      // Collect any potentially dead scalar instructions, including the
      // original scalars and pointer operands of loads/stores.
      if (NewVec != nullptr)
        collectPotentiallyDeadInstrs(Bndl);
      break;
    }
    case LegalityResultID::DiamondReuse: {
      NewVec = cast<DiamondReuse>(LegalityRes).getVector()->Vec;
      break;
    }
    case LegalityResultID::DiamondReuseWithShuffle: {
      auto *VecOp = cast<DiamondReuseWithShuffle>(LegalityRes).getVector()->Vec;
      const ShuffleMask &Mask =
          cast<DiamondReuseWithShuffle>(LegalityRes).getMask();
      NewVec = createShuffle(VecOp, Mask, UserBB);
      assert(NewVec->getType() == VecOp->getType() &&
             "Expected same type! Bad mask ?");
      break;
    }
    case LegalityResultID::DiamondReuseMultiInput: {
      const auto &Descr =
          cast<DiamondReuseMultiInput>(LegalityRes).getCollectDescr();
      Type *ResTy = VecUtils::getWideType(Bndl[0]->getType(), Bndl.size());

      // TODO: Try to get WhereIt without creating a vector.
      SmallVector<Value *, 4> DescrInstrs;
      for (const auto &ElmDescr : Descr.getDescrs()) {
        auto *V = ElmDescr.needsExtract() ? ElmDescr.getValue()->Vec
                                          : ElmDescr.getScalar();
        if (auto *I = dyn_cast<Instruction>(V))
          DescrInstrs.push_back(I);
      }
      BasicBlock::iterator WhereIt =
          getInsertPointAfterInstrs(DescrInstrs, UserBB);

      Value *LastV = PoisonValue::get(ResTy);
      Context &Ctx = LastV->getContext();
      unsigned Lane = 0;
      for (const auto &ElmDescr : Descr.getDescrs()) {
        Value *VecOp = nullptr;
        Value *ValueToInsert;
        if (ElmDescr.needsExtract()) {
          VecOp = ElmDescr.getValue()->Vec;
          ConstantInt *IdxC =
              ConstantInt::get(Type::getInt32Ty(Ctx), ElmDescr.getExtractIdx());
          ValueToInsert = ExtractElementInst::create(
              VecOp, IdxC, WhereIt, VecOp->getContext(), "VExt");
        } else {
          ValueToInsert = ElmDescr.getScalar();
        }
        auto NumLanesToInsert = VecUtils::getNumLanes(ValueToInsert);
        if (NumLanesToInsert == 1) {
          // If we are inserting a scalar element then we need a single insert.
          //   %VIns = insert %DstVec,  %SrcScalar, Lane
          ConstantInt *LaneC = ConstantInt::get(Type::getInt32Ty(Ctx), Lane);
          LastV = InsertElementInst::create(LastV, ValueToInsert, LaneC,
                                            WhereIt, Ctx, "VIns");
        } else {
          // If we are inserting a vector element then we need to extract and
          // insert each vector element one by one with a chain of extracts and
          // inserts, for example:
          //   %VExt0 = extract %SrcVec, 0
          //   %VIns0 = insert  %DstVec, %Vect0, Lane + 0
          //   %VExt1 = extract %SrcVec, 1
          //   %VIns1 = insert  %VIns0,  %Vect0, Lane + 1
          for (unsigned LnCnt = 0; LnCnt != NumLanesToInsert; ++LnCnt) {
            auto *ExtrIdxC = ConstantInt::get(Type::getInt32Ty(Ctx), LnCnt);
            auto *ExtrI = ExtractElementInst::create(ValueToInsert, ExtrIdxC,
                                                     WhereIt, Ctx, "VExt");
            unsigned InsLane = Lane + LnCnt;
            auto *InsLaneC = ConstantInt::get(Type::getInt32Ty(Ctx), InsLane);
            LastV = InsertElementInst::create(LastV, ExtrI, InsLaneC, WhereIt,
                                              Ctx, "VIns");
          }
        }
        Lane += NumLanesToInsert;
      }
      NewVec = LastV;
      break;
    }
    case LegalityResultID::Pack: {
      // If we can't vectorize the seeds then just return.
      if (Depth == 0)
        return nullptr;
      NewVec = createPack(Bndl, UserBB);
      break;
    }
    }
    if (NewVec != nullptr) {
      Change = true;
      ActionPtr->Vec = NewVec;
    }
#ifndef NDEBUG
    if (AlwaysVerify) {
      // This helps find broken IR by constantly verifying the function. Note
      // that this is very expensive and should only be used for debugging.
      Instruction *I0 = isa<Instruction>(Bndl[0])
                            ? cast<Instruction>(Bndl[0])
                            : cast<Instruction>(UserBndl[0]);
      assert(!Utils::verifyFunction(I0->getParent()->getParent(), dbgs()) &&
             "Broken function!");
    }
#endif // NDEBUG
  }
  return NewVec;
}

bool BottomUpVec::tryVectorize(ArrayRef<Value *> Bndl) {
  Change = false;
  if (LLVM_UNLIKELY(BottomUpInvocationCnt++ >= StopAt &&
                    StopAt != StopAtDisabled))
    return false;
  DeadInstrCandidates.clear();
  Legality->clear();
  Actions.clear();
  DebugBndlCnt = 0;
  vectorizeRec(Bndl, {}, /*Depth=*/0);
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "BottomUpVec: Vectorization Actions:\n";
             Actions.dump());
  emitVectors();
  tryEraseDeadInstrs();
  return Change;
}

bool BottomUpVec::runOnRegion(Region &Rgn, const Analyses &A) {
  const auto &SeedSlice = Rgn.getAux();
  assert(SeedSlice.size() >= 2 && "Bad slice!");
  Function &F = *SeedSlice[0]->getParent()->getParent();
  IMaps = std::make_unique<InstrMaps>();
  Legality = std::make_unique<LegalityAnalysis>(
      A.getAA(), A.getScalarEvolution(), F.getParent()->getDataLayout(),
      F.getContext(), *IMaps);

  // TODO: Refactor to remove the unnecessary copy to SeedSliceVals.
  SmallVector<Value *> SeedSliceVals(SeedSlice.begin(), SeedSlice.end());
  // Try to vectorize starting from the seed slice. The returned value
  // is true if we found vectorizable code and generated some vector
  // code for it. It does not mean that the code is profitable.
  return tryVectorize(SeedSliceVals);
}

} // namespace sandboxir
} // namespace llvm
